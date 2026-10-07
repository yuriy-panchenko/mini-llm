#pragma once
#include "afxdialogex.h"


// CNetSettingsDlg dialog

class CNetSettingsDlg : public CDialogEx
{
	DECLARE_DYNAMIC(CNetSettingsDlg)

public:
	CNetSettingsDlg(CWnd* pParent = nullptr);   // standard constructor
	virtual ~CNetSettingsDlg();

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_NET_SETTINGS_DLG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	afx_msg void OnChangeTextBrowse();
	afx_msg void OnChangeVocabBrowse();
	DECLARE_MESSAGE_MAP()

public:
	CString m_File_Vocab;
	CString m_File_Text;
	CString m_Seed;
	size_t m_Model;
	size_t m_Layers;
	size_t m_Heads;
	size_t m_CTX;
	double m_Lcoo;
	size_t m_VocabSize;
	std::string m_Corpus;
	llm::Tokenizer m_Tok;
};
