// CWorkerThread.cpp : implementation file
//

#include "pch.h"
#include "mini-llm-mfc.h"
#include "CWorkerThread.h"
#include "ChildView.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include "CNetSettingsDlg.h"

// CWorkerThread
using namespace llm;

IMPLEMENT_DYNCREATE(CWorkerThread, CWinThread)


CWorkerThread::CWorkerThread()
	:m_uStep{}
	, m_Rng{ std::random_device{}() }
	, m_Dist{ 0.f, 1.f }

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
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(m_Rng, m_Dist), Vector{dlg.m_Model} },
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(m_Rng, m_Dist), Vector{dlg.m_Model} },
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(m_Rng, m_Dist), Vector{dlg.m_Model} },
			Linear{ Matrix{dlg.m_Model, dlg.m_Model}.xavier(m_Rng, m_Dist), Vector{dlg.m_Model} },
			dlg.m_Heads, true };

		body.emplace_back(
			causalAttn,
			RMSNorm{ Vector{dlg.m_Model, 1.f} },
			Linear{ Matrix{dlg.m_Model, d_ff}.xavier(m_Rng, m_Dist), Vector{d_ff} },
			Linear{ Matrix{d_ff, dlg.m_Model}.xavier(m_Rng, m_Dist), Vector{dlg.m_Model} },
			RMSNorm{ Vector{dlg.m_Model, 1.f} });
	}

	// xavier() gives std sqrt(2/d_model) (~0.18 at d=64), which the sinusoidal PE (rms ~0.7)
	// swamps. Rescale to std ~1 so token identity and position start on the same scale.
	auto const embScale{ std::sqrt(2.f / static_cast<llm::scalar>(dlg.m_Model)) };
	m_Model = { Embedding{ Matrix{dlg.m_VocabSize, dlg.m_Model}.xavier(m_Rng, m_Dist) / embScale },
		std::move(body),Linear{ Matrix{dlg.m_Model, dlg.m_VocabSize}.xavier(m_Rng, m_Dist), Vector{dlg.m_VocabSize} } };

	m_Sample = CStringToUtf8(dlg.m_Seed);
	m_SampleIds = m_Tok.encode(m_Sample);
	m_AllIds = m_Tok.encode(dlg.m_Corpus);
	assert(m_AllIds.size() > dlg.m_CTX + 1);   // guards the subtraction below

	// Hold out the last 5% of the corpus (never trained on) for validation loss.
	// If either side would be shorter than one window, skip validation entirely.
	size_t const valLen{ m_AllIds.size() / 20 };
	m_TrainEnd = (valLen > m_CTX + 1 && m_AllIds.size() - valLen > m_CTX + 1)
		? m_AllIds.size() - valLen
		: m_AllIds.size();
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

std::vector<double> CWorkerThread::GetVal()
{
	std::lock_guard _o{ m_Mtx };
	std::vector<double> ret;
	ret.swap(m_Val);
	return ret;
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
	ASSERT(m_AllIds.size() > m_CTX + 1);

	size_t const start{ m_Rng() % (m_TrainEnd - m_CTX - 1) };   // train region only
	auto const itStart{ m_AllIds.begin() + start }, itEnd{ itStart + m_CTX };
	std::vector<size_t>
		x{ itStart , itEnd },
		y{ std::next(itStart), std::next(itEnd) };

	auto const logits{ m_Model.forward(x) };
	SetLoss(cross_entropy(logits, y));
	m_Model.zero_grad();
	m_Model.backward(cross_entropy_backward(logits, y));
	m_Model.update((llm::scalar)m_Lcoo);

	if (!(m_uStep % fast_mod))
	{
		SetText(TRUE, greedy_decode());
		m_pView->PostMessage(WM_FAST_FINISHED, m_uStep);
	}
	if (!(m_uStep % slow_mod))
	{
		if (auto const v{ EvalVal() }; !std::isnan(v))
		{
			std::lock_guard _o{ m_Mtx };
			m_Val.push_back(v);
		}
		SetText(FALSE, generate(m_Sample, 60));
		m_pView->PostMessage(WM_SLOW_FINISHED, m_uStep);
	}

	if (m_uStep < max_steps)
		PostThreadMessage(WM_NEXT_STEP, ++m_uStep, 0);
}

// Mean loss over a fixed set of evenly spaced held-out windows, so successive
// evaluations are comparable. NaN when there is no validation region.
double CWorkerThread::EvalVal()
{
	size_t const valLen{ m_AllIds.size() - m_TrainEnd };
	if (valLen <= m_CTX + 1)
		return std::numeric_limits<double>::quiet_NaN();

	size_t const slots{ valLen - m_CTX - 1 };
	size_t const n{ (std::min<size_t>)(16, slots) };
	double sum{};
	for (size_t i = 0; i < n; ++i)
	{
		auto const itStart{ m_AllIds.begin() + m_TrainEnd + i * slots / n }, itEnd{ itStart + m_CTX };
		std::vector<size_t> x{ itStart, itEnd }, y{ std::next(itStart), std::next(itEnd) };
		sum += cross_entropy(m_Model.forward(x), y);
	}
	return sum / static_cast<double>(n);
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

// Temperature + top-k sampling over one row of logits. Greedy argmax falls into the
// first repetition loop it finds; sampling shows what the model has actually learned.
template<typename It>
size_t CWorkerThread::sample_top_k(It first, It last)
{
	constexpr size_t topK{ 40 };
	constexpr llm::scalar temperature{ 0.8f };

	std::vector<llm::scalar> logit(first, last);
	std::vector<size_t> idx(logit.size());
	std::iota(idx.begin(), idx.end(), size_t{ 0 });

	auto const k{ (std::min)(topK, idx.size()) };
	std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
		[&](size_t a, size_t b) { return logit[a] > logit[b]; });

	auto const maxLogit{ logit[idx[0]] };
	std::vector<double> weight(k);
	for (size_t i = 0; i < k; ++i)
		weight[i] = std::exp((logit[idx[i]] - maxLogit) / temperature);

	std::discrete_distribution<size_t> pick{ weight.begin(), weight.end() };
	return idx[pick(m_Rng)];
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
		auto const next{ sample_top_k(row.begin(), row.end()) };
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
