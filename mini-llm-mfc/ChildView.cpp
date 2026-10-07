
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "mini-llm-mfc.h"
#include "ChildView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

constexpr int
chart_height{ 200 },
list_side_offset{ 5 };
// CChildView

CChildView::CChildView()
{}

void CChildView::Initialize(CNetSettingsDlg const& dlg)
{
}

CChildView::~CChildView()
{}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_WM_CREATE()
	ON_WM_SIZE()
END_MESSAGE_MAP()



// CChildView message handlers

BOOL CChildView::PreCreateWindow(CREATESTRUCT& cs)
{
	if (!CWnd::PreCreateWindow(cs))
		return FALSE;

	cs.dwExStyle |= WS_EX_CLIENTEDGE;
	cs.style &= ~WS_BORDER;
	cs.lpszClass = AfxRegisterWndClass(CS_HREDRAW | CS_VREDRAW | CS_DBLCLKS,
		::LoadCursor(nullptr, IDC_ARROW), reinterpret_cast<HBRUSH>(COLOR_WINDOW + 1), nullptr);
	cs.style |= WS_CLIPCHILDREN;

	return TRUE;
}

void CChildView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	// TODO: Add your message handler code here

	// Do not call CWnd::OnPaint() for painting messages
}


int CChildView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	if (!m_FastList.Create(WS_CHILD | WS_BORDER | WS_VISIBLE | LVS_REPORT | LVS_NOSORTHEADER | LVS_SHOWSELALWAYS | LVS_SINGLESEL, {}, this, ID_LIST_FAST))
		return -1;
	if (!m_SlowList.Create(WS_CHILD | WS_BORDER | WS_VISIBLE | LVS_REPORT | LVS_NOSORTHEADER | LVS_SHOWSELALWAYS | LVS_SINGLESEL, {}, this, ID_LIST_SLOW))
		return -1;

	int col{};
	m_FastList.SetExtendedStyle(LVS_EX_AUTOSIZECOLUMNS | LVS_EX_DOUBLEBUFFER | LVS_EX_FULLROWSELECT | LVS_EX_FULLROWSELECT);
	m_FastList.InsertColumn(col++, _T("N"));
	m_FastList.InsertColumn(col++, _T("Answer"));

	col = 0;
	m_SlowList.SetExtendedStyle(LVS_EX_AUTOSIZECOLUMNS | LVS_EX_DOUBLEBUFFER | LVS_EX_FULLROWSELECT | LVS_EX_FULLROWSELECT);
	m_SlowList.InsertColumn(col++, _T("N"));
	m_SlowList.InsertColumn(col++, _T("Answer"));

	return 0;
}

void CChildView::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);

	m_rChart = m_rCanvas = { 0, 0, cx, cy };
	m_rChart.bottom = m_rChart.top + chart_height;

	auto r{ m_rCanvas };
	r.top = m_rChart.bottom;
	r.right = m_rCanvas.CenterPoint().x;
	r.DeflateRect(list_side_offset, list_side_offset, list_side_offset, list_side_offset);
	m_FastList.MoveWindow(r);

	r = m_rCanvas;
	r.top = m_rChart.bottom;
	r.left = m_rCanvas.CenterPoint().x;
	r.DeflateRect(list_side_offset, list_side_offset, list_side_offset, list_side_offset);
	m_SlowList.MoveWindow(r);
}
