
// MainFrm.cpp : implementation of the CMainFrame class
//

#include "pch.h"
#include "framework.h"
#include "mini-llm-mfc.h"

#include "MainFrm.h"
#include "ChildFrm.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

// CMainFrame

IMPLEMENT_DYNAMIC(CMainFrame, CMDIFrameWnd)

BEGIN_MESSAGE_MAP(CMainFrame, CMDIFrameWnd)
	ON_WM_CREATE()
	ON_WM_CLOSE()
END_MESSAGE_MAP()

static UINT indicators[] =
{
	ID_SEPARATOR,           // status line indicator
	ID_INDICATOR_CAPS,
	ID_INDICATOR_NUM,
	ID_INDICATOR_SCRL,
};

// CMainFrame construction/destruction

CMainFrame::CMainFrame() noexcept
{
	// TODO: add member initialization code here
}

CMainFrame::~CMainFrame()
{}

int CMainFrame::OnCreate(LPCREATESTRUCT lpCreateStruct)
{
	if (CMDIFrameWnd::OnCreate(lpCreateStruct) == -1)
		return -1;

	if (!m_wndToolBar.CreateEx(this, TBSTYLE_FLAT, WS_CHILD | WS_VISIBLE | CBRS_TOP | CBRS_GRIPPER | CBRS_TOOLTIPS | CBRS_FLYBY | CBRS_SIZE_DYNAMIC) ||
		!m_wndToolBar.LoadToolBar(IDR_MAINFRAME))
	{
		TRACE0("Failed to create toolbar\n");
		return -1;      // fail to create
	}

	if (!m_wndStatusBar.Create(this))
	{
		TRACE0("Failed to create status bar\n");
		return -1;      // fail to create
	}
	m_wndStatusBar.SetIndicators(indicators, sizeof(indicators) / sizeof(UINT));

	// TODO: Delete these three lines if you don't want the toolbar to be dockable
	m_wndToolBar.EnableDocking(CBRS_ALIGN_ANY);
	EnableDocking(CBRS_ALIGN_ANY);
	DockControlBar(&m_wndToolBar);

	LoadSavedViews();

	return 0;
}

BOOL CMainFrame::PreCreateWindow(CREATESTRUCT& cs)
{
	if (!CMDIFrameWnd::PreCreateWindow(cs))
		return FALSE;
	// TODO: Modify the Window class or styles here by modifying
	//  the CREATESTRUCT cs

	return TRUE;
}

// CMainFrame diagnostics

#ifdef _DEBUG
void CMainFrame::AssertValid() const
{
	CMDIFrameWnd::AssertValid();
}

void CMainFrame::Dump(CDumpContext& dc) const
{
	CMDIFrameWnd::Dump(dc);
}
#endif //_DEBUG


// CMainFrame message handlers


void CMainFrame::OnClose()
{
	auto pAct{ static_cast<CChildFrame*>(CMDIFrameWnd::MDIGetActive()) }, pFirst{ pAct };
	int index{};
	CFile file;
	CString filename;
	if (pAct)do
	{
		filename.Format(_T("View%d.bin"), index);
		if (file.Open(filename, CFile::modeCreate | CFile::modeWrite | CFile::shareExclusive))
		{
			{
				CArchive ar{ &file, CArchive::store };
				pAct->GetView().Serialize(ar);
			}
			file.Close();
		}

		CMDIFrameWnd::MDINext();
		pAct = static_cast<CChildFrame*>(CMDIFrameWnd::MDIGetActive());
		++index;
	} while (pAct != pFirst);

	CMDIFrameWnd::OnClose();
}

void CMainFrame::LoadSavedViews()
{
	CString filename;
	CFile file;

	for (int i = 0; i < 20; ++i)
	{
		filename.Format(_T("View%d.bin"), i);
		if (file.Open(filename, CFile::modeRead | CFile::shareDenyWrite))
		{
			auto pFrame{ static_cast<CChildFrame*>(CreateNewChild(RUNTIME_CLASS(CChildFrame), IDR_minillmmfcTYPE, theApp.GetMDIMenu(), theApp.GetMDIAccel())) };
			{
				CArchive ar{ &file, CArchive::load };
				pFrame->GetView().Serialize(ar);
			}
			file.Close();
			CFile::Remove(filename);
		}
		else break;
	}
}
