
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "mini-llm-mfc.h"
#include "ChildView.h"
#include <algorithm>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

constexpr COLORREF token_colors[] =
{
	RGB(163,177,255), RGB(122,255,122), RGB(255,182,122), RGB(122,255,248),
	RGB(247,151,185), RGB(255,255,122), RGB(244,143,255), RGB(243,251,208),
	RGB(194,222,244), RGB(142,235,165), RGB(234,191,184), RGB(251,208,251),
	RGB(255,154,143), RGB(255,227,143), RGB(199,255,143), RGB(225,170,248),
	RGB(122,202,255), RGB(168,230,196), RGB(196,225,152), RGB(255,143,216),
	RGB(225,200,152), RGB(204,255,250), RGB(122,228,255), RGB(184,189,234),
};
constexpr int
chart_height{ 200 },
list_side_offset{ 5 };
// CChildView

CChildView::CChildView()
	:m_pTh{ nullptr }
	, m_bShowAll{ FALSE }
	, m_bShowMA50{ TRUE }
	, m_bShowZero{ FALSE }
	, m_bShowInfo{ TRUE }
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
	ON_WM_DRAWITEM()
	ON_COMMAND(ID_SHOW_INFO, &CChildView::OnShowInfo)
	ON_UPDATE_COMMAND_UI(ID_SHOW_INFO, &CChildView::OnUpdateShowInfo)
END_MESSAGE_MAP()



// CChildView message handlers

// Ctrl+Up / Ctrl+Down: learning rate x2 / x0.5; with Shift x10 / x0.1 (the "decay test": load a
// checkpoint, press Ctrl+Shift+Down once and watch Val). Works while a list has the focus.
BOOL CChildView::PreTranslateMessage(MSG* pMsg)
{
	if (pMsg->message == WM_KEYDOWN && m_pTh && (::GetKeyState(VK_CONTROL) & 0x8000)
		&& (pMsg->wParam == VK_UP || pMsg->wParam == VK_DOWN))
	{
		bool const up{ pMsg->wParam == VK_UP }, big{ (::GetKeyState(VK_SHIFT) & 0x8000) != 0 };
		ChangeLcoo(big ? (up ? 10. : .1) : (up ? 2. : .5));
		return TRUE;
	}
	return CWnd::PreTranslateMessage(pMsg);
}

void CChildView::ChangeLcoo(double factor)
{
	ASSERT(m_pTh);
	auto const lr{ (std::clamp)(m_pTh->GetLcoo() * factor, 1e-8, 1.0) };
	m_pTh->SetLcoo(lr);
	m_LrMarks.emplace_back(m_LossData.size(), lr);
	InvalidateRect(m_rChart);
}

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
			s.AppendFormat(_T("  MA50 %.4f"), m_MA50.back());
			if (!m_ValData.empty())
				s.AppendFormat(_T("  Val %.4f"), m_ValData.back());
			if (m_pTh)
				s.AppendFormat(_T("  LR %g"), m_pTh->GetLcoo());
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

			// Overlays share the raw curve's x/y mapping: white = MA50, green = validation.
			bool const scaled{ m_bShowAll && bCompressed };
			auto const from{ static_cast<size_t>(std::distance(m_LossData.cbegin(), itFrom)) };
			auto const toX{ [&](size_t i) {
				return scaled ? int(double(m_rChart.Width()) * (i - from) / m_LossData.size())
							  : int(i - from); } };
			auto const toY{ [&](double v) {
				auto const y{ int(m_rChart.Height() * (Max - v) / (Max - Min)) };
				return (std::max)(0, (std::min)(y, int(m_rChart.Height()) - 1)); } };

			if (m_bShowMA50)
			{
				CPen penMA{ PS_SOLID, 2, RGB(240, 240, 240) };
				memDC.SelectObject(penMA);
				memDC.MoveTo(toX(from), toY(m_MA50[from]));
				for (size_t i = from + 1; i < m_MA50.size(); ++i)
					memDC.LineTo(toX(i), toY(m_MA50[i]));
				memDC.SelectObject(pen);   // deselect before penMA is destroyed
			}

			if (!m_ValData.empty())
			{
				CPen penVal{ PS_SOLID, 2, RGB(60, 220, 60) };
				memDC.SelectObject(penVal);
				bool started{ false };
				for (size_t k = 0; k < m_ValData.size(); ++k)
				{
					auto const step{ k * slow_mod };
					if (step < from)
						continue;
					CPoint const p{ toX(step), toY(m_ValData[k]) };
					if (started)
						memDC.LineTo(p);
					else
						memDC.MoveTo(p);
					started = true;
				}
				memDC.SelectObject(pen);
			}

			// Learning-rate changes (Ctrl+Up/Down): dotted magenta line at that step, labelled with the new rate.
			if (!m_LrMarks.empty())
			{
				CPen penLr{ PS_DOT, 1, RGB(255, 80, 255) };
				memDC.SelectObject(penLr);
				memDC.SetTextColor(RGB(255, 120, 255));
				for (auto const& [idx, lr] : m_LrMarks)
				{
					if (idx < from || idx > m_LossData.size())
						continue;
					int const x{ toX(idx) };
					memDC.MoveTo(x, 0);
					memDC.LineTo(x, m_rChart.Height());
					CString t;
					t.Format(_T("%g"), lr);
					memDC.TextOut(x + 3, 22, t);
				}
				memDC.SelectObject(pen);   // deselect before penLr is destroyed
			}
		}
	}

	if (m_bShowInfo)
	{
		//memDC.DrawText(
	}

	dc.BitBlt(m_rChart.left, m_rChart.top, m_rChart.Width(), m_rChart.Height(), &memDC, 0, 0, SRCCOPY);
	memDC.RestoreDC(iSave);
}


int CChildView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	if (!m_FastList.Create(WS_CHILD | WS_BORDER | WS_VISIBLE
		| LVS_REPORT | LVS_NOSORTHEADER | LVS_SHOWSELALWAYS | LVS_SINGLESEL | LVS_OWNERDRAWFIXED,
		{}, this, ID_LIST_FAST))
		return -1;
	if (!m_SlowList.Create(WS_CHILD | WS_BORDER | WS_VISIBLE
		| LVS_REPORT | LVS_NOSORTHEADER | LVS_SHOWSELALWAYS | LVS_SINGLESEL | LVS_OWNERDRAWFIXED,
		{}, this, ID_LIST_SLOW))
		return -1;

	int col{};
	m_FastList.SetExtendedStyle(LVS_EX_AUTOSIZECOLUMNS | LVS_EX_DOUBLEBUFFER | LVS_EX_FULLROWSELECT | LVS_EX_FULLROWSELECT);
	m_FastList.InsertColumn(col++, _T("N"), 0, m_FastList.GetStringWidth(_T("999999")) + 16);
	m_FastList.InsertColumn(col++, _T("Answer"));

	col = 0;
	m_SlowList.SetExtendedStyle(LVS_EX_AUTOSIZECOLUMNS | LVS_EX_DOUBLEBUFFER | LVS_EX_FULLROWSELECT | LVS_EX_FULLROWSELECT);
	m_SlowList.InsertColumn(col++, _T("N"), 0, m_SlowList.GetStringWidth(_T("999999")) + 16);
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
		for (int i = 1; i < m_FastList.GetHeaderCtrl()->GetItemCount(); i++)
			m_FastList.SetColumnWidth(i, LVSCW_AUTOSIZE_USEHEADER);

	constexpr size_t maWindow{ 50 };
	for (double const l : m_pTh->GetLoss())
	{
		m_LossData.push_back(l);
		m_MA50Sum += l;
		auto const n{ m_LossData.size() };
		if (n > maWindow)
			m_MA50Sum -= m_LossData[n - maWindow - 1];
		m_MA50.push_back(m_MA50Sum / static_cast<double>((std::min)(n, maWindow)));
	}
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

	for (double const v : m_pTh->GetVal())
		m_ValData.push_back(v);

	++item.iSubItem;
	s = m_pTh->GetText(FALSE);
	s.Replace(_T("\n"), _T("\\n"));
	s.Replace(_T("\t"), _T("\\t"));
	item.pszText = (LPTSTR)(LPCTSTR)s;
	m_SlowList.SetItem(&item);

	if (should_show)
		m_SlowList.EnsureVisible(index, FALSE);

	if (!(wParam % (20 * slow_mod)))
		for (int i = 1; i < m_SlowList.GetHeaderCtrl()->GetItemCount(); i++)
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

void CChildView::OnDrawItem(int nIDCtl, LPDRAWITEMSTRUCT lpDrawItemStruct)
{
	if (m_pTh && nIDCtl == ID_LIST_FAST || nIDCtl == ID_LIST_SLOW)
	{
		CDC dc;
		if (!dc.Attach(lpDrawItemStruct->hDC))
			return;

		auto const iSave{ dc.SaveDC() };
		auto& list{ nIDCtl == ID_LIST_FAST ? m_FastList : m_SlowList };

		CString s, text{ list.GetItemText(lpDrawItemStruct->itemID, 0) };
		CRect const canvas{ lpDrawItemStruct->rcItem };
		auto rect{ canvas }, r{ rect };

		dc.SetBkMode(TRANSPARENT);
		rect.right = list.GetColumnWidth(0);
		r = rect;
		r.right -= 3;
		dc.DrawText(text, r, DT_SINGLELINE | DT_RIGHT | DT_VCENTER);

		text = list.GetItemText(lpDrawItemStruct->itemID, 1);
		auto const tokens{ m_pTh->Encode(CStringToUtf8(text)) };

		auto itToken{ tokens.begin() };
		dc.SetBkMode(OPAQUE);

		for (auto& cible : m_pTh->DecodeApart(tokens))
		{
			s = Utf8ToCString(cible);
			dc.DrawText(s, r, DT_SINGLELINE | DT_CALCRECT);

			rect.left = rect.right;
			rect.right += r.Width();

			dc.SetBkColor(nIDCtl == ID_LIST_SLOW
				? GetTokenColor(*itToken)
				: GetTokenColor(std::distance(tokens.begin(), itToken) % (sizeof(token_colors) / sizeof(COLORREF))));
			dc.DrawText(s, rect, DT_SINGLELINE);

			++itToken;
		}

		dc.RestoreDC(iSave);
		dc.Detach();
	}
}

COLORREF CChildView::GetTokenColor(llm::Tokenizer::TokenId id) const
{
	return token_colors[id % (sizeof(token_colors) / sizeof(COLORREF))];
}

void CChildView::OnShowInfo()
{
	m_bShowInfo = !m_bShowInfo;
	InvalidateRect(m_rChart);
}

void CChildView::OnUpdateShowInfo(CCmdUI* pCmdUI)
{
	pCmdUI->SetCheck(m_bShowInfo);
}

void CChildView::Serialize(CArchive& ar)
{
	if (ar.IsStoring())
	{
		//m_FastList.Serialize(ar);
		//m_SlowList.Serialize(ar);
		//CWorkerThread* m_pTh;
		//std::vector<double> m_LossData, m_MA50, m_ValData;   // m_MA50[i] pairs with m_LossData[i]; m_ValData[i] is step i*slow_mod
		//double m_MA50Sum{ 0 };
		//std::vector<std::pair<size_t, double>> m_LrMarks;   // (loss index, new learning rate) for every change
		//BOOL m_bShowAll, m_bShowMA50, m_bShowZero, m_bShowInfo;
	}
	else
	{	// loading code
	}
}
