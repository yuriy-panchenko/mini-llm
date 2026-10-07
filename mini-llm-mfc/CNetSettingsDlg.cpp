// CNetSettingsDlg.cpp : implementation file
//

#include "pch.h"
#include "mini-llm-mfc.h"
#include "afxdialogex.h"
#include "CNetSettingsDlg.h"
#include <fstream>

// CNetSettingsDlg dialog

IMPLEMENT_DYNAMIC(CNetSettingsDlg, CDialogEx)

CNetSettingsDlg::CNetSettingsDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_NET_SETTINGS_DLG, pParent)
	, m_File_Vocab(_T(""))
	, m_File_Text(_T(""))
	, m_Seed(_T(""))
	, m_Model{ 64 }
	, m_Layers{ 3 }
	, m_Heads{ 4 }
	, m_VocabSize{ 0 }
	, m_CTX{ 64 }
	, m_Lcoo{ .001 }
{
	m_Seed = _T("It appears to me");
}

CNetSettingsDlg::~CNetSettingsDlg()
{}

void CNetSettingsDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_VOCAB_BROWSE, m_File_Vocab);
	DDX_Text(pDX, IDC_TEXT_BROWSE, m_File_Text);
	DDX_Text(pDX, IDC_SEED_EDIT, m_Seed);
	DDX_Text(pDX, IDC_MODEL_EDIT, m_Model);
	DDX_Text(pDX, IDC_LAYERS_EDIT, m_Layers);
	DDX_Text(pDX, IDC_HEADS_EDIT, m_Heads);
	DDX_Text(pDX, IDC_CTX_EDIT, m_CTX);
	DDX_Text(pDX, IDC_LCOO_EDIT, m_Lcoo);
	DDX_Text(pDX, IDC_VOCAB_EDIT, m_VocabSize);
}


BEGIN_MESSAGE_MAP(CNetSettingsDlg, CDialogEx)
	ON_EN_CHANGE(IDC_TEXT_BROWSE, &CNetSettingsDlg::OnChangeTextBrowse)
	ON_EN_CHANGE(IDC_VOCAB_BROWSE, &CNetSettingsDlg::OnChangeVocabBrowse)
END_MESSAGE_MAP()


// CNetSettingsDlg message handlers

void CNetSettingsDlg::OnChangeTextBrowse()
{
	m_Corpus.clear();

	if (UpdateData())
	{
		std::ifstream file(CStringToUtf8(m_File_Text));
		if (file)
			m_Corpus = llm::normalize_whitespace(std::string{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() });
	}
}

void CNetSettingsDlg::OnChangeVocabBrowse()
{
	if (UpdateData())
	{
		std::ifstream file{ CStringToUtf8(m_File_Vocab), std::ios::binary };
		if (file)
		{
			file >> m_Tok;
			m_VocabSize = m_Tok.vocab_size();
			UpdateData(FALSE);
		}
	}
}
