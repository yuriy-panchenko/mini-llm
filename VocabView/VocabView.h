
// VocabView.h : main header file for the VocabView application
//
#pragma once

#ifndef __AFXWIN_H__
	#error "include 'pch.h' before including this file for PCH"
#endif

#include "resource.h"       // main symbols


// CVocabViewApp:
// See VocabView.cpp for the implementation of this class
//

class CVocabViewApp : public CWinApp
{
public:
	CVocabViewApp() noexcept;


// Overrides
public:
	virtual BOOL InitInstance();
	virtual int ExitInstance();

// Implementation

public:
	afx_msg void OnAppAbout();
	DECLARE_MESSAGE_MAP()
};

extern CVocabViewApp theApp;
