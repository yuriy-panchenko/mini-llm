
// mini-llm-mfc.h : main header file for the mini-llm-mfc application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CMiniLLMApp:
// See mini-llm-mfc.cpp for the implementation of this class
//
std::string CStringToUtf8(CString const& s);
CString Utf8ToCString(std::string const& s);

class CMiniLLMApp : public CWinApp
{
public:
	CMiniLLMApp() noexcept;


// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation
protected:
	HMENU  m_hMDIMenu;
	HACCEL m_hMDIAccel;

public:
	afx_msg void OnAppAbout();
	afx_msg void OnFileNew();
	DECLARE_MESSAGE_MAP()
};

extern CMiniLLMApp theApp;
