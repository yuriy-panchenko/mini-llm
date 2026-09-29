
// ChildView.h : interface of the CChildView class
//

#pragma once
#include "VocabModel.h"

// CChildView window. Two toolbar rows on top; below them either a flat sortable LIST or a
// merge-hierarchy TREE (switch with the "View" button), and on the right the statistics and
// token-detail panes. All controls are created in code (OnCreate): no .rc / Resource.h changes.

class CChildView : public CWnd
{
// Construction
public:
	CChildView();

// Operations
public:
	bool LoadFile(CString const& path);		// false (after showing a message) if the file can't be loaded

// Overrides
protected:
	virtual BOOL PreCreateWindow(CREATESTRUCT& cs);

// Implementation
public:
	virtual ~CChildView();

protected:
	afx_msg void OnPaint();
	afx_msg int  OnCreate(LPCREATESTRUCT lpCreateStruct);
	afx_msg void OnSize(UINT nType, int cx, int cy);
	afx_msg void OnSetFocus(CWnd* pOldWnd);

	afx_msg void OnLoad();
	afx_msg void OnToggleView();
	afx_msg void OnBack();
	afx_msg void OnParentA();
	afx_msg void OnParentB();
	afx_msg void OnGo();
	afx_msg void OnFind();
	afx_msg void OnSortCombo();
	afx_msg void OnSortOrder();

	afx_msg void OnGetDispInfo(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnColumnClick(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnItemChanged(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnTreeSelChanged(NMHDR* pNMHDR, LRESULT* pResult);
	afx_msg void OnTreeExpanding(NMHDR* pNMHDR, LRESULT* pResult);
	DECLARE_MESSAGE_MAP()

private:
	bool TreeMode() const;
	void Layout(int cx, int cy);
	void ApplySort(int column, bool ascending);
	void UpdateSortUi();
	void UpdateSortArrows();

	void SelectToken(uint32_t id);				// selects in whichever view is showing
	void OnTokenSelected(uint32_t id);			// common handler: history, detail pane, buttons
	int  RowOf(uint32_t id) const;

	void RebuildTree();
	HTREEITEM InsertNode(HTREEITEM parent, uint32_t id);
	void SelectTokenInTree(uint32_t id);
	CString NodeLabel(uint32_t id) const;

	void ShowDetail(uint32_t id);
	void ShowReport(CString const& path, unsigned fileVersion);
	void UpdateButtons();
	void SetText(CEdit& edit, std::string const& utf8);

	vv::VocabModel m_model;
	uint32_t m_current{ vv::kNone };			// token id currently selected
	std::vector<uint32_t> m_history;			// previously selected ids, for Back
	bool m_navBack{ false };					// true while Back is re-selecting (don't record history)
	bool m_rebuilding{ false };					// true while the tree is being torn down/rebuilt (ignore notifications)
	bool m_treeDirty{ true };					// tree is stale (new file or new sort order)
	int  m_sortCol{ 0 };
	bool m_sortAsc{ true };

	CFont m_fontMono;
	CButton m_btnLoad, m_btnView, m_btnBack, m_btnParentA, m_btnParentB, m_btnOrder, m_btnGo, m_btnFind;
	CStatic m_stSort, m_stGoto, m_stFind;
	CComboBox m_cbSort;
	CEdit m_edGoto, m_edFind;
	CListCtrl m_list;
	CTreeCtrl m_tree;
	CEdit m_edStats, m_edDetail;
};

