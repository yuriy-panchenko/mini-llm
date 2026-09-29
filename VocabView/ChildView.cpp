
// ChildView.cpp : implementation of the CChildView class
//

#include "pch.h"
#include "framework.h"
#include "ChildView.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

namespace
{
	enum : UINT
	{
		IDC_LOAD = 2001, IDC_VIEW, IDC_BACK, IDC_PARENT_A, IDC_PARENT_B,
		IDC_SORT_LABEL, IDC_SORT, IDC_ORDER,
		IDC_GOTO_LABEL, IDC_GOTO, IDC_GO, IDC_FIND_LABEL, IDC_FIND_EDIT, IDC_FIND,
		IDC_LIST, IDC_TREE, IDC_STATS, IDC_DETAIL
	};

	struct ColumnDef
	{
		TCHAR const* name;
		int width;
		int fmt;
		vv::Column col;
		bool descFirst;			// first sort on this column is largest-first (more useful for counts)
	};
	constexpr ColumnDef kCols[]
	{
		{ _T("Id"),      60,  LVCFMT_LEFT,  vv::Column::Id,      false },
		{ _T("Merge #"), 70,  LVCFMT_RIGHT, vv::Column::Rank,    false },
		{ _T("Bytes"),   55,  LVCFMT_RIGHT, vv::Column::Bytes,   true  },
		{ _T("Depth"),   55,  LVCFMT_RIGHT, vv::Column::Depth,   true  },
		{ _T("Uses"),    55,  LVCFMT_RIGHT, vv::Column::Uses,    true  },
		{ _T("Freq"),    80,  LVCFMT_RIGHT, vv::Column::Freq,    true  },
		{ _T("Parents"), 100, LVCFMT_LEFT,  vv::Column::Parents, false },
		{ _T("Text"),    280, LVCFMT_LEFT,  vv::Column::Text,    false },
	};
	constexpr int kNumCols{ int(sizeof kCols / sizeof kCols[0]) };
	constexpr size_t kMaxHistory{ 200 };
	constexpr int kBtnH{ 24 };

	CString Utf8ToCString(std::string const& s)
	{
		CString out;
		if (s.empty()) return out;
		int const n{ MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0) };
		if (n <= 0) return out;
		MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), out.GetBuffer(n), n);
		out.ReleaseBuffer(n);
		return out;
	}

	std::string CStringToUtf8(CString const& s)
	{
		if (s.IsEmpty()) return {};
		int const n{ WideCharToMultiByte(CP_UTF8, 0, s.GetString(), s.GetLength(), nullptr, 0, nullptr, nullptr) };
		std::string out((size_t)(n > 0 ? n : 0), '\0');
		if (n > 0) WideCharToMultiByte(CP_UTF8, 0, s.GetString(), s.GetLength(), out.data(), n, nullptr, nullptr);
		return out;
	}
}

// CChildView

CChildView::CChildView()
{}

CChildView::~CChildView()
{}


BEGIN_MESSAGE_MAP(CChildView, CWnd)
	ON_WM_PAINT()
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_SETFOCUS()
	ON_BN_CLICKED(IDC_LOAD, &CChildView::OnLoad)
	ON_BN_CLICKED(IDC_VIEW, &CChildView::OnToggleView)
	ON_BN_CLICKED(IDC_BACK, &CChildView::OnBack)
	ON_BN_CLICKED(IDC_PARENT_A, &CChildView::OnParentA)
	ON_BN_CLICKED(IDC_PARENT_B, &CChildView::OnParentB)
	ON_BN_CLICKED(IDC_ORDER, &CChildView::OnSortOrder)
	ON_BN_CLICKED(IDC_GO, &CChildView::OnGo)
	ON_BN_CLICKED(IDC_FIND, &CChildView::OnFind)
	ON_CBN_SELCHANGE(IDC_SORT, &CChildView::OnSortCombo)
	ON_NOTIFY(LVN_GETDISPINFO, IDC_LIST, &CChildView::OnGetDispInfo)
	ON_NOTIFY(LVN_COLUMNCLICK, IDC_LIST, &CChildView::OnColumnClick)
	ON_NOTIFY(LVN_ITEMCHANGED, IDC_LIST, &CChildView::OnItemChanged)
	ON_NOTIFY(TVN_SELCHANGED, IDC_TREE, &CChildView::OnTreeSelChanged)
	ON_NOTIFY(TVN_ITEMEXPANDING, IDC_TREE, &CChildView::OnTreeExpanding)
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

	return TRUE;
}

void CChildView::OnPaint()
{
	CPaintDC dc(this); // device context for painting

	// Children cover the client area; nothing to paint here.
}

int CChildView::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	CRect const r0{ 0, 0, 0, 0 };
	DWORD const vis{ WS_CHILD | WS_VISIBLE };
	CFont* const pGui{ CFont::FromHandle((HFONT)::GetStockObject(DEFAULT_GUI_FONT)) };
	m_fontMono.CreatePointFont(90, _T("Consolas"));

	bool ok{ true };
	ok &= !!m_btnLoad.Create(_T("Load..."), vis | WS_TABSTOP | BS_PUSHBUTTON, r0, this, IDC_LOAD);
	ok &= !!m_btnView.Create(_T("View: List"), vis | WS_TABSTOP | BS_AUTOCHECKBOX | BS_PUSHLIKE, r0, this, IDC_VIEW);
	ok &= !!m_btnBack.Create(_T("< Back"), vis | WS_TABSTOP | BS_PUSHBUTTON | WS_DISABLED, r0, this, IDC_BACK);
	ok &= !!m_btnParentA.Create(_T("Parent A"), vis | WS_TABSTOP | BS_PUSHBUTTON | WS_DISABLED, r0, this, IDC_PARENT_A);
	ok &= !!m_btnParentB.Create(_T("Parent B"), vis | WS_TABSTOP | BS_PUSHBUTTON | WS_DISABLED, r0, this, IDC_PARENT_B);
	ok &= !!m_stSort.Create(_T("Sort"), vis | SS_CENTERIMAGE | SS_RIGHT, r0, this, IDC_SORT_LABEL);
	ok &= !!m_cbSort.Create(vis | WS_TABSTOP | WS_VSCROLL | CBS_DROPDOWNLIST, r0, this, IDC_SORT);
	ok &= !!m_btnOrder.Create(_T("Asc"), vis | WS_TABSTOP | BS_PUSHBUTTON, r0, this, IDC_ORDER);
	ok &= !!m_stGoto.Create(_T("Go to #"), vis | SS_CENTERIMAGE | SS_RIGHT, r0, this, IDC_GOTO_LABEL);
	ok &= !!m_edGoto.Create(vis | WS_TABSTOP | WS_BORDER | ES_AUTOHSCROLL, r0, this, IDC_GOTO);
	ok &= !!m_btnGo.Create(_T("Go"), vis | WS_TABSTOP | BS_PUSHBUTTON, r0, this, IDC_GO);
	ok &= !!m_stFind.Create(_T("Find"), vis | SS_CENTERIMAGE | SS_RIGHT, r0, this, IDC_FIND_LABEL);
	ok &= !!m_edFind.Create(vis | WS_TABSTOP | WS_BORDER | ES_AUTOHSCROLL, r0, this, IDC_FIND_EDIT);
	ok &= !!m_btnFind.Create(_T("Next"), vis | WS_TABSTOP | BS_PUSHBUTTON, r0, this, IDC_FIND);

	ok &= !!m_list.Create(vis | WS_TABSTOP | WS_BORDER | LVS_REPORT | LVS_OWNERDATA | LVS_SINGLESEL | LVS_SHOWSELALWAYS,
		r0, this, IDC_LIST);
	// The tree starts hidden; the View button swaps it with the list.
	ok &= !!m_tree.Create(WS_CHILD | WS_TABSTOP | WS_BORDER | TVS_HASBUTTONS | TVS_HASLINES | TVS_LINESATROOT | TVS_SHOWSELALWAYS,
		r0, this, IDC_TREE);
	DWORD const editStyle{ vis | WS_BORDER | WS_VSCROLL | WS_HSCROLL | ES_MULTILINE | ES_READONLY | ES_AUTOVSCROLL | ES_AUTOHSCROLL };
	ok &= !!m_edStats.Create(editStyle, r0, this, IDC_STATS);
	ok &= !!m_edDetail.Create(editStyle, r0, this, IDC_DETAIL);
	if (!ok)
	{
		TRACE0("Failed to create VocabView controls\n");
		return -1;
	}

	for (CWnd* w : { (CWnd*)&m_btnLoad, (CWnd*)&m_btnView, (CWnd*)&m_btnBack, (CWnd*)&m_btnParentA, (CWnd*)&m_btnParentB,
		(CWnd*)&m_stSort, (CWnd*)&m_cbSort, (CWnd*)&m_btnOrder, (CWnd*)&m_stGoto, (CWnd*)&m_edGoto, (CWnd*)&m_btnGo,
		(CWnd*)&m_stFind, (CWnd*)&m_edFind, (CWnd*)&m_btnFind })
		w->SetFont(pGui);
	m_list.SetFont(&m_fontMono);
	m_tree.SetFont(&m_fontMono);
	m_edStats.SetFont(&m_fontMono);
	m_edDetail.SetFont(&m_fontMono);
	m_edStats.SetLimitText(1 << 20);
	m_edDetail.SetLimitText(1 << 20);

	m_list.SetExtendedStyle(LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER | LVS_EX_HEADERDRAGDROP);
	for (int i = 0; i < kNumCols; ++i)
	{
		m_list.InsertColumn(i, kCols[i].name, kCols[i].fmt, kCols[i].width);
		m_cbSort.AddString(kCols[i].name);
	}
	UpdateSortUi();

	SetText(m_edStats, "Press Load... to open a tokenizer file (*.tok / *.bin).");
	return 0;
}

void CChildView::OnSize(UINT nType, int cx, int cy)
{
	CWnd::OnSize(nType, cx, cy);
	if (m_list.GetSafeHwnd())
		Layout(cx, cy);
}

void CChildView::OnSetFocus(CWnd* /*pOldWnd*/)
{
	if (TreeMode())
		m_tree.SetFocus();
	else
		m_list.SetFocus();
}

bool CChildView::TreeMode() const
{
	return m_btnView.GetSafeHwnd() && m_btnView.GetCheck() == BST_CHECKED;
}

void CChildView::Layout(int cx, int cy)
{
	int const m{ 6 }, bh{ kBtnH };

	// Row 1: Load | View | Back | Parent A | Parent B | Sort [combo] [Asc/Desc]
	int x{ m };
	auto place = [&](CWnd& w, int y, int width, int height = kBtnH) { w.MoveWindow(x, y, width, height); x += width + m; };
	place(m_btnLoad, m, 70);
	place(m_btnView, m, 90);
	place(m_btnBack, m, 60);
	place(m_btnParentA, m, 70);
	place(m_btnParentB, m, 70);
	x += 12;
	place(m_stSort, m, 34);
	place(m_cbSort, m, 100, 220);				// height of a combo = height of its drop-down list
	place(m_btnOrder, m, 50);

	// Row 2: Go to # [edit] Go | Find [edit] Next
	int const y2{ m + bh + m };
	x = m;
	place(m_stGoto, y2, 50);
	place(m_edGoto, y2, 60);
	place(m_btnGo, y2, 34);
	x += 12;
	place(m_stFind, y2, 34);
	place(m_edFind, y2, 150);
	place(m_btnFind, y2, 50);

	int const top{ y2 + bh + m };
	int const height{ (std::max)(0, cy - top - m) };
	int const leftW{ (std::max)(200, (cx - 3 * m) * 6 / 10) };
	m_list.MoveWindow(m, top, leftW, height);
	m_tree.MoveWindow(m, top, leftW, height);

	int const rx{ m + leftW + m };
	int const rw{ (std::max)(0, cx - rx - m) };
	int const statsH{ (std::max)(0, (height - m) * 55 / 100) };
	m_edStats.MoveWindow(rx, top, rw, statsH);
	m_edDetail.MoveWindow(rx, top + statsH + m, rw, (std::max)(0, height - statsH - m));
}

// ---- loading ----------------------------------------------------------------------------------

void CChildView::OnLoad()
{
	CFileDialog dlg(TRUE, _T("tok"), nullptr, OFN_FILEMUSTEXIST | OFN_HIDEREADONLY | OFN_PATHMUSTEXIST,
		_T("Tokenizer files (*.tok;*.bin)|*.tok;*.bin|All files (*.*)|*.*||"), this);
	if (dlg.DoModal() == IDOK)
		LoadFile(dlg.GetPathName());
}

bool CChildView::LoadFile(CString const& path)
{
	auto fail = [&](CString const& why)
		{
			AfxMessageBox(_T("Cannot load \"") + path + _T("\":\n\n") + why, MB_ICONWARNING);
			return false;
		};

	CWaitCursor wait;		// files that carry the token stream can be large

	std::ifstream f{ std::filesystem::path{ path.GetString() }, std::ios::binary };
	if (!f)
		return fail(_T("The file could not be opened."));

	vv::VocabModel loaded;
	{
		// Tokenizer's deserializer validates the magic/version/counts itself and throws on a bad file.
		llm::Tokenizer tok;
		try
		{
			f >> tok;
		}
		catch (std::exception const& e)
		{
			return fail(CString(e.what()));
		}
		catch (...)
		{
			return fail(_T("The data is corrupt."));
		}
		f.peek();
		if (!f.eof())
			return fail(_T("The file has unexpected data after the tokenizer."));

		std::string err;
		if (!loaded.build(tok.vocab(), tok.merges(), &err))
			return fail(Utf8ToCString(err));

		loaded.set_corpus(tok.get_tokens());	// only counts are kept; the stream itself is dropped with `tok`
		std::vector<vv::PairCount> pairs;
		pairs.reserve(tok.pair_counts().size());
		for (auto const& [p, c] : tok.pair_counts())
			pairs.push_back({ p, (uint64_t)c });
		loaded.set_pairs(std::move(pairs));
	}
	loaded.sort(kCols[m_sortCol].col, m_sortAsc);		// keep the user's current sort across files

	// Success: swap in the new model and reset the view.
	m_list.SetItemState(-1, 0, LVIS_SELECTED | LVIS_FOCUSED);
	m_model = std::move(loaded);
	m_current = vv::kNone;
	m_history.clear();
	m_treeDirty = true;
	m_list.SetItemCount((int)m_model.size());
	m_list.Invalidate();
	if (TreeMode())
		RebuildTree();

	ShowReport(path, 2);
	SelectToken(0);
	UpdateButtons();

	CString const name{ std::filesystem::path{ path.GetString() }.filename().c_str() };
	if (CWnd* pMain = AfxGetMainWnd())
		pMain->SetWindowText(name + _T(" - VocabView"));
	return true;
}

// ---- view switch ------------------------------------------------------------------------------

void CChildView::OnToggleView()
{
	bool const tree{ TreeMode() };
	m_btnView.SetWindowText(tree ? _T("View: Tree") : _T("View: List"));
	m_list.ShowWindow(tree ? SW_HIDE : SW_SHOW);
	m_tree.ShowWindow(tree ? SW_SHOW : SW_HIDE);

	if (tree && m_treeDirty)
		RebuildTree();								// also selects m_current
	else if (m_current != vv::kNone)
		SelectToken(m_current);						// bring the newly shown view to the current token

	if (tree) m_tree.SetFocus(); else m_list.SetFocus();
}

// ---- list callbacks ---------------------------------------------------------------------------

void CChildView::OnGetDispInfo(NMHDR* pNMHDR, LRESULT* pResult)
{
	*pResult = 0;
	auto* pDI{ reinterpret_cast<NMLVDISPINFO*>(pNMHDR) };
	LVITEM& item{ pDI->item };
	if (!(item.mask & LVIF_TEXT))
		return;
	auto const& order{ m_model.order() };
	if (item.iItem < 0 || (size_t)item.iItem >= order.size())
		return;

	auto const& t{ m_model.info(order[item.iItem]) };
	CString s;
	switch (item.iSubItem)
	{
	case 0: s.Format(_T("%u"), t.id); break;
	case 1: if (t.rank != vv::kNone) s.Format(_T("%u"), t.rank); else s = _T("-"); break;
	case 2: s.Format(_T("%u"), t.byteLen); break;
	case 3: s.Format(_T("%u"), t.depth); break;
	case 4: s.Format(_T("%u"), t.uses); break;
	case 5: if (m_model.hasCorpus()) s.Format(_T("%llu"), t.freq); else s = _T("-"); break;
	case 6: if (t.parentA != vv::kNone) s.Format(_T("%u + %u"), t.parentA, t.parentB); else s = _T("-"); break;
	case 7: s = Utf8ToCString(t.display); break;
	}
	lstrcpyn(item.pszText, s, item.cchTextMax);
}

void CChildView::OnColumnClick(NMHDR* pNMHDR, LRESULT* pResult)
{
	*pResult = 0;
	auto* p{ reinterpret_cast<NMLISTVIEW*>(pNMHDR) };
	int const c{ p->iSubItem };
	if (c < 0 || c >= kNumCols)
		return;
	bool const asc{ c == m_sortCol ? !m_sortAsc : !kCols[c].descFirst };
	ApplySort(c, asc);
}

void CChildView::OnItemChanged(NMHDR* pNMHDR, LRESULT* pResult)
{
	*pResult = 0;
	auto* p{ reinterpret_cast<NMLISTVIEW*>(pNMHDR) };
	if (p->iItem < 0 || !(p->uChanged & LVIF_STATE))
		return;
	if (!(p->uNewState & LVIS_SELECTED) || (p->uOldState & LVIS_SELECTED))
		return;						// only react to "became selected"
	auto const& order{ m_model.order() };
	if ((size_t)p->iItem >= order.size())
		return;
	OnTokenSelected(order[p->iItem]);
}

// ---- sorting (shared by the list header, the Sort combo and the Asc/Desc button) --------------

void CChildView::OnSortCombo()
{
	int const c{ m_cbSort.GetCurSel() };
	if (c >= 0 && c < kNumCols && c != m_sortCol)
		ApplySort(c, !kCols[c].descFirst);
}

void CChildView::OnSortOrder()
{
	ApplySort(m_sortCol, !m_sortAsc);
}

void CChildView::ApplySort(int column, bool ascending)
{
	m_sortCol = column;
	m_sortAsc = ascending;
	UpdateSortUi();
	if (m_model.size() == 0)
		return;

	m_model.sort(kCols[column].col, ascending);
	m_treeDirty = true;

	if (TreeMode())
	{
		RebuildTree();								// children lists follow the new order
		return;
	}

	// Owner-data selection is by row, so after re-ordering, re-select the same *token*.
	uint32_t const keep{ m_current };
	m_list.SetItemState(-1, 0, LVIS_SELECTED | LVIS_FOCUSED);
	m_list.Invalidate();
	if (keep != vv::kNone)
		SelectToken(keep);
}

void CChildView::UpdateSortUi()
{
	m_cbSort.SetCurSel(m_sortCol);
	m_btnOrder.SetWindowText(m_sortAsc ? _T("Asc") : _T("Desc"));
	UpdateSortArrows();
}

void CChildView::UpdateSortArrows()
{
	CHeaderCtrl* const hdr{ m_list.GetHeaderCtrl() };
	if (!hdr)
		return;
	for (int i = 0; i < kNumCols; ++i)
	{
		HDITEM h{};
		h.mask = HDI_FORMAT;
		hdr->GetItem(i, &h);
		h.fmt &= ~(HDF_SORTUP | HDF_SORTDOWN);
		if (i == m_sortCol)
			h.fmt |= m_sortAsc ? HDF_SORTUP : HDF_SORTDOWN;
		hdr->SetItem(i, &h);
	}
}

// ---- tree -------------------------------------------------------------------------------------
//
// The tree is the merge DAG rooted at the 256 base bytes: a node's children are the tokens built
// directly from it. Children are inserted lazily the first time a node is expanded. A token has
// two parents, so it appears under both of them (selecting either copy selects the same token).

CString CChildView::NodeLabel(uint32_t id) const
{
	auto const& t{ m_model.info(id) };
	CString s;
	s.Format(_T("#%u  \"%s\"  %uB"), t.id, (LPCTSTR)Utf8ToCString(t.display), t.byteLen);
	if (m_model.hasCorpus())
	{
		CString f;
		f.Format(_T("  x%llu"), t.freq);
		s += f;
	}
	return s;
}

HTREEITEM CChildView::InsertNode(HTREEITEM parent, uint32_t id)
{
	CString const label{ NodeLabel(id) };
	TVINSERTSTRUCT tvi{};
	tvi.hParent = parent;
	tvi.hInsertAfter = TVI_LAST;
	tvi.item.mask = TVIF_TEXT | TVIF_PARAM | TVIF_CHILDREN;
	tvi.item.pszText = const_cast<LPTSTR>(label.GetString());
	tvi.item.lParam = (LPARAM)id;
	tvi.item.cChildren = m_model.info(id).uses > 0 ? 1 : 0;		// shows the [+] before children exist
	return m_tree.InsertItem(&tvi);
}

void CChildView::RebuildTree()
{
	m_rebuilding = true;
	m_tree.SetRedraw(FALSE);
	m_tree.DeleteAllItems();
	if (m_model.size() > 0)
		for (uint32_t id : m_model.roots())
			InsertNode(TVI_ROOT, id);
	m_tree.SetRedraw(TRUE);
	m_rebuilding = false;
	m_treeDirty = false;

	if (m_current != vv::kNone)
		SelectTokenInTree(m_current);
}

void CChildView::OnTreeExpanding(NMHDR* pNMHDR, LRESULT* pResult)
{
	*pResult = 0;
	if (m_rebuilding)
		return;
	auto* p{ reinterpret_cast<NMTREEVIEW*>(pNMHDR) };
	if (p->action != TVE_EXPAND || !p->itemNew.hItem)
		return;
	HTREEITEM const h{ p->itemNew.hItem };
	if (m_tree.GetChildItem(h) != nullptr)
		return;											// already populated
	uint32_t const id{ (uint32_t)m_tree.GetItemData(h) };
	for (uint32_t child : m_model.children(id))
		InsertNode(h, child);
}

void CChildView::OnTreeSelChanged(NMHDR* pNMHDR, LRESULT* pResult)
{
	*pResult = 0;
	if (m_rebuilding)
		return;
	auto* p{ reinterpret_cast<NMTREEVIEW*>(pNMHDR) };
	if (!p->itemNew.hItem)
		return;
	OnTokenSelected((uint32_t)m_tree.GetItemData(p->itemNew.hItem));
}

void CChildView::SelectTokenInTree(uint32_t id)
{
	if (id >= m_model.size())
		return;

	// Path from a base byte down to `id`: follow parentA (left parent) upward; every token is a
	// child of its left parent, so the reversed chain is a valid root-to-token path.
	std::vector<uint32_t> path;
	for (uint32_t t = id;;)
	{
		path.push_back(t);
		uint32_t const pa{ m_model.info(t).parentA };
		if (pa == vv::kNone || pa >= t)
			break;
		t = pa;
	}
	std::reverse(path.begin(), path.end());

	HTREEITEM h{ m_tree.GetRootItem() };
	while (h && (uint32_t)m_tree.GetItemData(h) != path[0])
		h = m_tree.GetNextSiblingItem(h);
	for (size_t i = 1; h && i < path.size(); ++i)
	{
		m_tree.Expand(h, TVE_EXPAND);					// populates the children via OnTreeExpanding
		HTREEITEM c{ m_tree.GetChildItem(h) };
		while (c && (uint32_t)m_tree.GetItemData(c) != path[i])
			c = m_tree.GetNextSiblingItem(c);
		h = c;
	}
	if (!h)
	{
		::MessageBeep(MB_ICONASTERISK);				// e.g. a token of a corrupt file that isn't reachable from the roots
		return;
	}
	m_tree.SelectItem(h);
	m_tree.EnsureVisible(h);
}

// ---- selection & navigation -------------------------------------------------------------------

int CChildView::RowOf(uint32_t id) const
{
	auto const& order{ m_model.order() };
	auto const it{ std::find(order.begin(), order.end(), id) };
	return it == order.end() ? -1 : int(it - order.begin());
}

void CChildView::SelectToken(uint32_t id)
{
	if (TreeMode())
	{
		if (m_treeDirty)
			RebuildTree();
		SelectTokenInTree(id);
		return;
	}

	int const row{ RowOf(id) };
	if (row < 0)
	{
		::MessageBeep(MB_ICONASTERISK);
		return;
	}
	m_list.SetItemState(-1, 0, LVIS_SELECTED);
	m_list.SetItemState(row, LVIS_SELECTED | LVIS_FOCUSED, LVIS_SELECTED | LVIS_FOCUSED);
	m_list.EnsureVisible(row, FALSE);
}

void CChildView::OnTokenSelected(uint32_t id)
{
	if (id >= m_model.size())
		return;
	if (!m_navBack && m_current != vv::kNone && m_current != id)
	{
		m_history.push_back(m_current);
		if (m_history.size() > kMaxHistory)
			m_history.erase(m_history.begin());
	}
	m_current = id;
	ShowDetail(id);
	UpdateButtons();
}

void CChildView::OnBack()
{
	if (m_history.empty())
		return;
	uint32_t const id{ m_history.back() };
	m_history.pop_back();
	m_navBack = true;
	SelectToken(id);
	m_navBack = false;
	UpdateButtons();
}

void CChildView::OnParentA()
{
	if (m_current != vv::kNone)
		SelectToken(m_model.info(m_current).parentA);
}

void CChildView::OnParentB()
{
	if (m_current != vv::kNone)
		SelectToken(m_model.info(m_current).parentB);
}

void CChildView::OnGo()
{
	CString s;
	m_edGoto.GetWindowText(s);
	s.Trim();
	if (!s.IsEmpty() && s[0] == _T('#'))
		s = s.Mid(1);
	bool digits{ !s.IsEmpty() && s.GetLength() <= 6 };
	for (int i = 0; i < s.GetLength(); ++i)
		digits &= (s[i] >= _T('0') && s[i] <= _T('9'));
	if (!digits)
	{
		::MessageBeep(MB_ICONASTERISK);
		return;
	}
	SelectToken((uint32_t)_ttoi(s));
}

void CChildView::OnFind()
{
	CString s;
	m_edFind.GetWindowText(s);
	std::string needle{ CStringToUtf8(s) };
	// The views show a space as a middle dot; accept the dot in the search box too.
	for (size_t pos = 0; (pos = needle.find("\xC2\xB7", pos)) != std::string::npos;)
		needle.replace(pos, 2, " ");
	uint32_t const hit{ m_model.find(needle, m_current) };
	if (hit == vv::kNone)
		::MessageBeep(MB_ICONASTERISK);
	else
		SelectToken(hit);
}

// ---- text panes -------------------------------------------------------------------------------

void CChildView::SetText(CEdit& edit, std::string const& utf8)
{
	CString w{ Utf8ToCString(utf8) };
	w.Replace(_T("\n"), _T("\r\n"));
	edit.SetWindowText(w);
}

void CChildView::ShowReport(CString const& path, unsigned fileVersion)
{
	std::string head{ "File: " + CStringToUtf8(path) + "\n" };
	head += fileVersion >= 2 ? "Format: v" + std::to_string(fileVersion) + " (magic MTOK)\n\n"
		: std::string{ "Format: v1 (legacy, no magic; vocabulary and merges only)\n\n" };
	SetText(m_edStats, head + m_model.report());
}

void CChildView::UpdateButtons()
{
	bool const merged{ m_current != vv::kNone && m_model.info(m_current).parentA != vv::kNone };
	m_btnBack.EnableWindow(!m_history.empty());
	m_btnParentA.EnableWindow(merged);
	m_btnParentB.EnableWindow(merged);
}

void CChildView::ShowDetail(uint32_t id)
{
	auto const& t{ m_model.info(id) };
	auto const& raw{ m_model.bytes(id) };
	char buf[160];
	std::string s;

	snprintf(buf, sizeof buf, "Token #%u\n", t.id);
	s += buf;
	s += "Text:  \"" + t.display + "\"\n";

	s += "Hex:   ";
	for (size_t i = 0; i < raw.size() && i < 32; ++i)
	{
		snprintf(buf, sizeof buf, "%02X ", (unsigned)(unsigned char)raw[i]);
		s += buf;
	}
	s += raw.size() > 32 ? "...\n" : "\n";

	snprintf(buf, sizeof buf, "Bytes: %u    Depth: %u    Used as parent in: %u merge(s)\n", t.byteLen, t.depth, t.uses);
	s += buf;
	if (m_model.hasCorpus())
	{
		snprintf(buf, sizeof buf, "Occurrences in saved stream: %llu (%.3f%% of %llu tokens)\n",
			(unsigned long long)t.freq, 100.0 * t.freq / m_model.corpusTokens(), (unsigned long long)m_model.corpusTokens());
		s += buf;
	}
	snprintf(buf, sizeof buf, "UTF-8: %s    Starts with space: %s    Has newline: %s\n",
		t.validUtf8 ? "valid" : "partial/invalid", t.startsWithSpace ? "yes" : "no", t.hasNewline ? "yes" : "no");
	s += buf;

	if (t.parentA == vv::kNone)
		s += "\nBase byte (no merge).\n";
	else
	{
		snprintf(buf, sizeof buf, "\nMerge #%u = #%u + #%u\n", t.rank, t.parentA, t.parentB);
		s += buf;
		if (t.parentA < m_model.size() && t.parentB < m_model.size())
			s += "  \"" + m_model.info(t.parentA).display + "\" + \"" + m_model.info(t.parentB).display + "\"\n";
		if (!t.consistent)
			s += "  WARNING: bytes are not the concatenation of the parents (corrupt/mismatched file).\n";
		s += "Tree: " + m_model.tree(id) + "\n";
	}

	auto const kids{ m_model.children(id) };
	snprintf(buf, sizeof buf, "\nDirect children (%zu, in current sort order):\n", kids.size());
	s += buf;
	for (size_t i = 0; i < kids.size() && i < 40; ++i)
	{
		snprintf(buf, sizeof buf, "  #%u  ", kids[i]);
		s += buf;
		s += "\"" + m_model.info(kids[i]).display + "\"";
		if (m_model.hasCorpus())
		{
			snprintf(buf, sizeof buf, "  x%llu", (unsigned long long)m_model.info(kids[i]).freq);
			s += buf;
		}
		s += "\n";
	}
	if (kids.size() > 40)
	{
		snprintf(buf, sizeof buf, "  ... and %zu more\n", kids.size() - 40);
		s += buf;
	}

	SetText(m_edDetail, s);
}
