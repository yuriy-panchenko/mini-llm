
// ChildView.h : interface of the CChildView class
//


#pragma once
#include "CWorkerThread.h"
class CNetSettingsDlg;

// CChildView window
#define WM_FAST_FINISHED		(WM_USER+0x0001)
#define WM_SLOW_FINISHED		(WM_USER+0x0002)

class CChildView : public CWnd
{
	// Construction
public:
	CChildView();
	void Initialize(CNetSettingsDlg const&);
	// Attributes
public:

	// Operations
public:

	// Overrides
protected:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

	// Implementation
public:
	virtual ~CChildView();

	// Generated message map functions
protected:
	afx_msg void OnPaint();
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnDestroy();
	afx_msg LRESULT OnFastText(WPARAM, LPARAM);
	afx_msg LRESULT OnSlowText(WPARAM, LPARAM);
	DECLARE_MESSAGE_MAP()

private:
	CRect m_rCanvas, m_rChart;
	CListCtrl m_FastList, m_SlowList;
	CWorkerThread* m_pTh;
};

