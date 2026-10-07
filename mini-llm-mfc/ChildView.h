
// ChildView.h : interface of the CChildView class
//


#pragma once
class CNetSettingsDlg;

constexpr size_t fast_mod{ 10ull }, slow_mod{ 100ull };
// CChildView window

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
	DECLARE_MESSAGE_MAP()

private:
	CRect m_rCanvas, m_rChart;
	CListCtrl m_FastList, m_SlowList;
};

