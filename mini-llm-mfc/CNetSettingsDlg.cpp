// CNetSettingsDlg.cpp : implementation file
//

#include "pch.h"
#include "mini-llm-mfc.h"
#include "afxdialogex.h"
#include "CNetSettingsDlg.h"


// CNetSettingsDlg dialog

IMPLEMENT_DYNAMIC(CNetSettingsDlg, CDialogEx)

CNetSettingsDlg::CNetSettingsDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_NET_SETTINGS_DLG, pParent)
	, m_File_Vocab(_T(""))
	, m_File_Text(_T(""))
	, m_Seed(_T(""))
	, m_Lcoo(0)
{

}

CNetSettingsDlg::~CNetSettingsDlg()
{
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
END_MESSAGE_MAP()


// CNetSettingsDlg message handlers
