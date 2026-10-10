#pragma once
#include <mutex>
// CWorkerThread
class CNetSettingsDlg;
class CChildView;

#define WM_NEXT_STEP	(WM_APP+0x0001)
constexpr size_t fast_mod{ 10ull }, slow_mod{ 100ull }, max_steps{ 100'000ull };

class CWorkerThread : public CWinThread
{
	DECLARE_DYNCREATE(CWorkerThread)

protected:
	CWorkerThread();           // protected constructor used by dynamic creation
	virtual ~CWorkerThread();

public:
	void Init(CChildView* pView, CNetSettingsDlg const&);
	CString GetText(BOOL isFast);
	std::vector<double> GetLoss();
	std::vector<double> GetVal();   // validation loss, one entry per slow_mod steps

public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

protected:
	afx_msg void OnNextStep(WPARAM, LPARAM);
	DECLARE_MESSAGE_MAP()

private:
	std::string greedy_decode();
	double EvalVal();
	std::string generate(std::string const& prompt, size_t maxNewTokens);
	template<typename It> size_t sample_top_k(It first, It last);
	void SetText(BOOL isFast, std::string&&);
	void SetLoss(double);

private:
	CChildView* m_pView;
	llm::Tokenizer m_Tok;
	llm::Model<llm::MultiHeadAttention> m_Model;
	std::vector<llm::Tokenizer::TokenId> m_AllIds, m_SampleIds;
	size_t m_CTX, m_uStep;
	double m_Lcoo;
	std::string m_Sample;

	std::mutex m_Mtx;
	CString m_FastText, m_SlowText;
	std::vector<double> m_Loss, m_Val;
	size_t m_TrainEnd{ 0 };   // ids [m_TrainEnd, end) are held out and never trained on

	std::mt19937 m_Rng;
	std::normal_distribution<llm::scalar> m_Dist;
};


