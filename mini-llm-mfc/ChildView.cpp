
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
	:m_pTh{ nullptr }
	, m_bShowAll{ FALSE }
	, m_bShowMA50{ TRUE }
	, m_bShowZero{ TRUE }
{}

void CChildView::Initialize(CNetSettingsDlg const& dlg)
{
	ASSERT(!m_pTh);
	if (m_pTh = static_cast<CWorkerThread*>(::AfxBeginThread(RUNTIME_CLASS(CWorkerThread), 0, 0, CREATE_SUSPENDED)))
	{
		m_pTh->Init(this, dlg);
		m_pTh->ResumeThread();
	}
}

CChildView::~CChildView()
{}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_DESTROY()
	ON_MESSAGE(WM_FAST_FINISHED, &OnFastText)
	ON_MESSAGE(WM_SLOW_FINISHED, &OnSlowText)
	ON_COMMAND(ID_SHOW_ALL, &CChildView::OnShowAll)
	ON_UPDATE_COMMAND_UI(ID_SHOW_ALL, &CChildView::OnUpdateShowAll)
	ON_COMMAND(ID_SHOW_MOVING_AVERAGE, &CChildView::OnShowMovingAverage)
	ON_UPDATE_COMMAND_UI(ID_SHOW_MOVING_AVERAGE, &CChildView::OnUpdateShowMovingAverage)
	ON_WM_ERASEBKGND()
	ON_COMMAND(ID_SHOW_ZERO, &CChildView::OnShowZero)
	ON_UPDATE_COMMAND_UI(ID_SHOW_ZERO, &CChildView::OnUpdateShowZero)
	ON_WM_CONTEXTMENU()
	ON_COMMAND(ID_EDIT_COPY, &CChildView::OnEditCopy)
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

	CBitmap bmp;
	bmp.CreateCompatibleBitmap(&dc, m_rChart.Width(), m_rChart.Height());
	CDC memDC;
	memDC.CreateCompatibleDC(&dc);
	int iSave{ memDC.SaveDC() };
	memDC.SelectObject(bmp);
	CPen pen{ PS_SOLID,1,RGB(200,200,20) };
	memDC.SelectObject(pen);

	if (!m_LossData.empty())
	{
		auto itFrom{ m_LossData.cbegin() }, itTo{ m_LossData.cend() };
		bool const bCompressed{ m_LossData.size() > m_rChart.Width() };
		if (!m_bShowAll && bCompressed)
			itFrom = itTo - m_rChart.Width();

		double Max{ *itFrom }, Min{ Max };
		for (auto it{ std::next(itFrom) }; it != itTo; ++it)
		{
			Max = max(Max, *it);
			Min = min(Min, *it);
		}
		if (Max > Min)
		{
			if (m_bShowZero)
				Min = .0;
			CString s;
			s.Format(_T("Max %.4f  Min %.4f  Count %I64u"), Max, Min, m_LossData.size());
			CFont font;
			font.CreatePointFont(90, _T("Consolas"));
			memDC.SelectObject(font);
			memDC.SetTextColor(RGB(200, 200, 200));
			memDC.SetBkMode(TRANSPARENT);
			memDC.DrawText(s, CRect{ 0,5,m_rChart.Width(),m_rChart.Height() }, DT_SINGLELINE | DT_TOP | DT_CENTER);

			CPoint pnt{};
			if (m_bShowAll && bCompressed)
			{
				pnt.y = int(m_rChart.Height() * (Max - *itFrom) / (Max - Min));
				memDC.MoveTo(pnt);
				for (auto it{ std::next(itFrom) }; it != itTo; ++it)
				{
					pnt.x = int(double(m_rChart.Width()) * std::distance(itFrom, it) / m_LossData.size());
					pnt.y = int(m_rChart.Height() * (Max - *it) / (Max - Min));
					memDC.LineTo(pnt);
				}
			}
			else
			{
				pnt.y = int(m_rChart.Height() * (Max - *itFrom) / (Max - Min));
				memDC.MoveTo(pnt);
				for (auto it{ std::next(itFrom) }; it != itTo; ++it)
				{
					++pnt.x;
					pnt.y = int(m_rChart.Height() * (Max - *it) / (Max - Min));
					memDC.LineTo(pnt);
				}
			}
		}
	}

	dc.BitBlt(m_rChart.left, m_rChart.top, m_rChart.Width(), m_rChart.Height(), &memDC, 0, 0, SRCCOPY);
	memDC.RestoreDC(iSave);
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
	r.right = m_rCanvas.left + m_rCanvas.Width() * 1 / 4;
	r.DeflateRect(list_side_offset, list_side_offset, list_side_offset, list_side_offset);
	m_FastList.MoveWindow(r);

	r = m_rCanvas;
	r.top = m_rChart.bottom;
	r.left = m_rCanvas.left + m_rCanvas.Width() * 1 / 4;
	r.DeflateRect(list_side_offset, list_side_offset, list_side_offset, list_side_offset);
	m_SlowList.MoveWindow(r);
}

void CChildView::OnDestroy()
{
	CWnd::OnDestroy();

	if (m_pTh)
	{
		m_pTh->PostThreadMessage(WM_QUIT, 0, 0);
		::WaitForSingleObject(*m_pTh, INFINITE);
		m_pTh = nullptr;
	}
}

LRESULT CChildView::OnFastText(WPARAM wParam, LPARAM)
{
	ASSERT(m_pTh);

	CString s;
	LVITEM item{};
	item.mask = LVIF_TEXT;
	item.iItem = m_FastList.GetItemCount();
	bool const should_show{ item.iItem ? (bool)m_FastList.IsItemVisible(item.iItem - 1) : true };
	s.Format(_T("%I64u"), wParam);
	item.pszText = (LPTSTR)(LPCTSTR)s;
	auto const index{ m_FastList.InsertItem(&item) };

	++item.iSubItem;
	s = m_pTh->GetText(TRUE);
	s.Replace(_T("\n"), _T("\\n"));
	s.Replace(_T("\t"), _T("\\t"));
	item.pszText = (LPTSTR)(LPCTSTR)s;
	m_FastList.SetItem(&item);

	if (should_show)
		m_FastList.EnsureVisible(index, FALSE);

	if (!(wParam % (20 * fast_mod)))
		for (int i = 0; i < m_FastList.GetHeaderCtrl()->GetItemCount(); i++)
			m_FastList.SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);

	auto loss{ m_pTh->GetLoss() };
	m_LossData.insert(m_LossData.end(), loss.begin(), loss.end());
	InvalidateRect(m_rChart);

	return 0;
}

LRESULT CChildView::OnSlowText(WPARAM wParam, LPARAM)
{
	ASSERT(m_pTh);

	CString s;
	LVITEM item{};
	item.mask = LVIF_TEXT;
	item.iItem = m_SlowList.GetItemCount();
	bool const should_show{ item.iItem ? (bool)m_SlowList.IsItemVisible(item.iItem - 1) : true };
	s.Format(_T("%I64u"), wParam);
	item.pszText = (LPTSTR)(LPCTSTR)s;
	auto const index{ m_SlowList.InsertItem(&item) };

	++item.iSubItem;
	s = m_pTh->GetText(FALSE);
	s.Replace(_T("\n"), _T("\\n"));
	s.Replace(_T("\t"), _T("\\t"));
	item.pszText = (LPTSTR)(LPCTSTR)s;
	m_SlowList.SetItem(&item);

	if (should_show)
		m_SlowList.EnsureVisible(index, FALSE);

	if (!(wParam % (20 * slow_mod)))
		for (int i = 0; i < m_SlowList.GetHeaderCtrl()->GetItemCount(); i++)
			m_SlowList.SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);

	return 0;
}

void CChildView::OnShowAll()
{
	m_bShowAll = !m_bShowAll;
	InvalidateRect(m_rChart);
}

void CChildView::OnUpdateShowAll(CCmdUI* pCmdUI)
{
	pCmdUI->SetCheck(m_bShowAll);
}

void CChildView::OnShowMovingAverage()
{
	m_bShowMA50 = !m_bShowMA50;
	InvalidateRect(m_rChart);
}

void CChildView::OnUpdateShowMovingAverage(CCmdUI* pCmdUI)
{
	pCmdUI->SetCheck(m_bShowMA50);
}

BOOL CChildView::OnEraseBkgnd(CDC* pDC)
{
	// Erase everything EXCEPT the chart.
	CRect rClient;
	GetClientRect(&rClient);
	auto iSave{ pDC->SaveDC() };

	CBrush* pOldBrush = (CBrush*)pDC->SelectStockObject(WHITE_BRUSH);

	// This is the important part.
	pDC->ExcludeClipRect(m_rChart);

	pDC->FillRect(&rClient, (CBrush*)CBrush::FromHandle(
		(HBRUSH)GetStockObject(WHITE_BRUSH)));

	pDC->RestoreDC(iSave);
	//pDC->SelectObject(pOldBrush);

	return TRUE;    // We handled the background erase.
}
void CChildView::OnShowZero()
{
	m_bShowZero = !m_bShowZero;
	InvalidateRect(m_rChart);
}

void CChildView::OnUpdateShowZero(CCmdUI* pCmdUI)
{
	pCmdUI->SetCheck(m_bShowZero);
}

void CChildView::OnContextMenu(CWnd* pWnd, CPoint point)
{
	if (pWnd == &m_SlowList)
	{
		CMenu menu;
		if (menu.LoadMenu(IDR_COPY_MENU))
		{
			auto pMenu{ menu.GetSubMenu(0) };
			//auto pMenu{ &menu };
			if (pMenu)
				pMenu->TrackPopupMenu(0, point.x, point.y, this);
		}
	}
}

void CChildView::OnEditCopy()
{
	CopyListCtrlToClipboard(m_SlowList);
}
