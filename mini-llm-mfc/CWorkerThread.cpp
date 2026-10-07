// CWorkerThread.cpp : implementation file
//

#include "pch.h"
#include "mini-llm-mfc.h"
#include "CWorkerThread.h"
#include "ChildView.h"
#include <fstream>
#include "CNetSettingsDlg.h"

// CWorkerThread
using namespace llm;

IMPLEMENT_DYNCREATE(CWorkerThread, CWinThread)

std::mt19937 CWorkerThread::g_Rng{ std::random_device{}() };
std::normal_distribution<scalar> CWorkerThread::g_Dist{ 0.f, 1.f };

CWorkerThread::CWorkerThread()
	:m_uStep{}
{}

CWorkerThread::~CWorkerThread()
{}

void CWorkerThread::Init(CChildView* pView, CNetSettingsDlg const& dlg)
{
	m_pView = pView;
	m_CTX = dlg.m_CTX;
	m_uStep = 0;
	m_Lcoo = dlg.m_Lcoo;
	m_Tok = dlg.m_Tok;

	std::vector<llm::TransformerBlock<llm::MultiHeadAttention>> body;
	body.reserve(dlg.m_Layers);
	size_t const d_ff{ dlg.m_Model << 2 };   // classic 4× expansion

	for (size_t i = 0; i < dlg.m_Layers; ++i)
	{
		// Fresh weights per layer -- constructing this outside the loop and
		// copying it into every emplace_back() would give every layer (and every
		// one of Wq/Wk/Wv/Wo within a layer) byte-identical starting weights.
		MultiHeadAttention causalAttn{
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(g_Rng, g_Dist), Vector{dlg.m_Model}.random(g_Rng, g_Dist) },
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(g_Rng, g_Dist), Vector{dlg.m_Model}.random(g_Rng, g_Dist) },
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(g_Rng, g_Dist), Vector{dlg.m_Model}.random(g_Rng, g_Dist) },
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(g_Rng, g_Dist), Vector{dlg.m_Model}.random(g_Rng, g_Dist) },
			dlg.m_Heads, true };

		body.emplace_back(
			causalAttn,
			RMSNorm{ Vector{dlg.m_Model, 1.f} },
			Linear{ Matrix{dlg.m_Model, d_ff}.xavier(g_Rng, g_Dist), Vector{d_ff}.random(g_Rng, g_Dist) },
			Linear{ Matrix{d_ff, dlg.m_Model}.xavier(g_Rng, g_Dist), Vector{dlg.m_Model}.random(g_Rng, g_Dist) },
			RMSNorm{ Vector{dlg.m_Model, 1.f} });
	}

	m_Model = { Embedding{ Matrix{dlg.m_VocabSize, dlg.m_Model}.xavier(g_Rng, g_Dist) },
		std::move(body),Linear{ Matrix{dlg.m_Model, dlg.m_VocabSize}.xavier(g_Rng, g_Dist), Vector{dlg.m_VocabSize} } };

	m_Sample = CStringToUtf8(dlg.m_Seed);
	m_SampleIds = m_Tok.encode(m_Sample);
	m_AllIds = m_Tok.encode(dlg.m_Corpus);
	assert(m_AllIds.size() > dlg.m_CTX + 1);   // guards the subtraction below
}

CString CWorkerThread::GetText(BOOL isFast)
{
	std::lock_guard _o{ m_Mtx };
	return isFast ? m_FastText : m_SlowText;
}

std::vector<double> CWorkerThread::GetLoss()
{
	std::lock_guard _o{ m_Mtx };
	return std::move(m_Loss);
}

BOOL CWorkerThread::InitInstance()
{
	PostThreadMessage(WM_NEXT_STEP, 0, 0);
	return TRUE;
}

int CWorkerThread::ExitInstance()
{
	// TODO:  perform any per-thread cleanup here
	return CWinThread::ExitInstance();
}

BEGIN_MESSAGE_MAP(CWorkerThread, CWinThread)
	ON_THREAD_MESSAGE(WM_NEXT_STEP, &OnNextStep)
END_MESSAGE_MAP()


// CWorkerThread message handlers
void CWorkerThread::OnNextStep(WPARAM, LPARAM)
{
	ASSERT(m_pView);

	size_t const start{ g_Rng() % (m_AllIds.size() - m_CTX - 1) };

	std::vector<size_t>
		x{ m_AllIds.begin() + start, m_AllIds.begin() + start + m_CTX },
		y{ m_AllIds.begin() + start + 1, m_AllIds.begin() + start + m_CTX + 1 };

	auto const logits{ m_Model.forward(x) };
	SetLoss(cross_entropy(logits, y));
	m_Model.zero_grad();
	auto const dLogits{ cross_entropy_backward(logits, y) };
	m_Model.backward(dLogits);
	m_Model.update((llm::scalar)m_Lcoo);

	if (!(m_uStep % fast_mod))
	{
		SetText(TRUE, greedy_decode());
		m_pView->PostMessage(WM_FAST_FINISHED, m_uStep);
	}
	if (!(m_uStep % slow_mod))
	{
		SetText(FALSE, generate(m_Sample, 60));
		m_pView->PostMessage(WM_SLOW_FINISHED, m_uStep);
	}

	if (m_uStep < max_steps)
		PostThreadMessage(WM_NEXT_STEP, ++m_uStep, 0);
}

std::string CWorkerThread::greedy_decode()
{
	auto logits{ m_Model.forward({ m_SampleIds.begin(), m_SampleIds.end() }) };

	std::vector<Tokenizer::TokenId> predicted;
	predicted.reserve(logits.rows());
	for (size_t r = 0; r < logits.rows(); ++r)
	{
		auto row{ logits.row(r) };

		auto itMax{ std::max_element(row.begin(), row.end()) };
		predicted.push_back((Tokenizer::TokenId)std::distance(row.begin(), itMax));
	}
	return m_Tok.decode(predicted);
}

std::string CWorkerThread::generate(std::string const& prompt, size_t maxNewTokens)
{
	std::vector<size_t> window;
	for (auto id : m_Tok.encode(prompt))
		window.push_back(id);

	std::string ret;
	for (size_t i = 0; i < maxNewTokens; ++i)
	{
		std::vector<size_t> in(window.size() > m_CTX ? window.end() - m_CTX : window.begin(), window.end());
		auto logits{ m_Model.forward(in) };
		auto row{ logits.row(logits.rows() - 1) };
		auto next{ static_cast<size_t>(std::distance(row.begin(), std::max_element(row.begin(), row.end()))) };
		window.push_back(next);
		ret += m_Tok.decode({ static_cast<Tokenizer::TokenId>(next) });
	}
	return ret;
}

void CWorkerThread::SetText(BOOL isFast, std::string&& str)
{
	std::lock_guard _o{ m_Mtx };
	(isFast ? m_FastText : m_SlowText) = Utf8ToCString(str);
}

void CWorkerThread::SetLoss(double val)
{
	std::lock_guard _o{ m_Mtx };
	m_Loss.push_back(val);
}
