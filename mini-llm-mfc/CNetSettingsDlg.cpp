// CNetSettingsDlg.cpp : implementation file
//

#include "pch.h"
#include "mini-llm-mfc.h"
#include "afxdialogex.h"
#include "CNetSettingsDlg.h"
#include <fstream>

// CNetSettingsDlg dialog
#define SETTINGS		_T("Settings")
#define KEY_VOCAB	_T("VocabFile")
#define KEY_TEXT		_T("TextFile")
#define KEY_SEED		_T("Seed")
#define KEY_MODEL	_T("Model")
#define KEY_LAYERS	_T("Layers")
#define KEY_HEADS	_T("Heads")
#define KEY_CTX		_T("CTX")
#define KEY_VOCAB_SIZE		_T("VocabSize")
#define KEY_LCOO		_T("Lcoo")

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
	m_File_Vocab = theApp.GetProfileString(SETTINGS, KEY_VOCAB);
	m_File_Text = theApp.GetProfileString(SETTINGS, KEY_TEXT);
	m_Seed = theApp.GetProfileString(SETTINGS, KEY_SEED);

	m_Model = theApp.GetProfileInt(SETTINGS, KEY_MODEL, 64);
	m_Layers = theApp.GetProfileInt(SETTINGS, KEY_LAYERS, 2);
	m_Heads = theApp.GetProfileInt(SETTINGS, KEY_HEADS, 4);
	m_CTX = theApp.GetProfileInt(SETTINGS, KEY_CTX, 64);
	m_VocabSize = theApp.GetProfileInt(SETTINGS, KEY_VOCAB_SIZE, 0);

	auto s = theApp.GetProfileString(SETTINGS, KEY_LCOO, _T(".001"));
	m_Lcoo = _wtof(s);
}

CNetSettingsDlg::~CNetSettingsDlg()
{
	theApp.WriteProfileString(SETTINGS, KEY_VOCAB, m_File_Vocab);
	theApp.WriteProfileString(SETTINGS, KEY_TEXT, m_File_Text);
	theApp.WriteProfileString(SETTINGS, KEY_SEED, m_Seed);

	theApp.WriteProfileInt(SETTINGS, KEY_MODEL, (int)m_Model);
	theApp.WriteProfileInt(SETTINGS, KEY_LAYERS, (int)m_Layers);
	theApp.WriteProfileInt(SETTINGS, KEY_HEADS, (int)m_Heads);
	theApp.WriteProfileInt(SETTINGS, KEY_CTX, (int)m_CTX);
	theApp.WriteProfileInt(SETTINGS, KEY_VOCAB_SIZE, (int)m_VocabSize);

	CString s;
	s.Format(_T("%f"), m_Lcoo);
	theApp.WriteProfileString(SETTINGS, KEY_LCOO, s);
}

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
			//m_Corpus = llm::normalize_whitespace(std::string{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() });
			m_Corpus = std::string{ std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>() };
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

BOOL CNetSettingsDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();


	if (!m_File_Text.IsEmpty())
		OnChangeTextBrowse();
	if (!m_File_Vocab.IsEmpty())
		OnChangeVocabBrowse();

	return TRUE;  // return TRUE unless you set the focus to a control
	// EXCEPTION: OCX Property Pages should return FALSE
}
