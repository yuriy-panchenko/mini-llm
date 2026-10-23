
// ChildFrm.h : interface of the CChildFrame class
//

#pragma once
#include "ChildView.h"

class CChildFrame : public CMDIChildWnd
{
	DECLARE_DYNCREATE(CChildFrame)
public:
	CChildFrame() noexcept;

	auto& GetView()const { return m_wndView; };
	auto& GetView() { return m_wndView; };
	int GetNumber() const { return m_nNumber; }   // 1, 2, 3... in creation order; used in the title and the chart file names

	// Attributes
protected:
	CSplitterWnd m_wndSplitter;
	int m_nNumber{ 0 };
public:

	// Operations
public:

	// Overrides
public:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
	virtual BOOL OnCmdMsg(UINT nID, int nCode, void* pExtra, AFX_CMDHANDLERINFO* pHandlerInfo);

	// Implementation
public:
	// view for the client area of the frame.
	CChildView m_wndView;
	virtual ~CChildFrame();
#ifdef _DEBUG
	virtual void AssertValid() const;
	virtual void Dump(CDumpContext& dc) const;
#endif

	// Generated message map functions
protected:
	afx_msg void OnFileClose();
	afx_msg void OnSetFocus(CWnd* pOldWnd);
	afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
	DECLARE_MESSAGE_MAP()
public:
	virtual BOOL DestroyWindow();
	virtual void PostNcDestroy();
};
