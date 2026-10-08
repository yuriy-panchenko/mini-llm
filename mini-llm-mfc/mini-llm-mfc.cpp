
// mini-llm-mfc.cpp : Defines the class behaviors for the application.
//

#include "pch.h"
#include "framework.h"
#include "afxwinappex.h"
#include "afxdialogex.h"
#include "mini-llm-mfc.h"
#include "MainFrm.h"

#include "ChildFrm.h"
#include "CNetSettingsDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// CMiniLLMApp

BEGIN_MESSAGE_MAP(CMiniLLMApp, CWinApp)
	ON_COMMAND(ID_APP_ABOUT, &CMiniLLMApp::OnAppAbout)
	ON_COMMAND(ID_FILE_NEW, &CMiniLLMApp::OnFileNew)
END_MESSAGE_MAP()


// CMiniLLMApp construction

CMiniLLMApp::CMiniLLMApp() noexcept
{

	// support Restart Manager
	m_dwRestartManagerSupportFlags = AFX_RESTART_MANAGER_SUPPORT_ALL_ASPECTS;
#ifdef _MANAGED
	// If the application is built using Common Language Runtime support (/clr):
	//     1) This additional setting is needed for Restart Manager support to work properly.
	//     2) In your project, you must add a reference to System.Windows.Forms in order to build.
	System::Windows::Forms::Application::SetUnhandledExceptionMode(System::Windows::Forms::UnhandledExceptionMode::ThrowException);
#endif

	// TODO: replace application ID string below with unique ID string; recommended
	// format for string is CompanyName.ProductName.SubProduct.VersionInformation
	SetAppID(_T("minillmmfc.AppID.NoVersion"));

	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}

// The one and only CMiniLLMApp object

CMiniLLMApp theApp;


// CMiniLLMApp initialization

BOOL CMiniLLMApp::InitInstance()
{
	// InitCommonControlsEx() is required on Windows XP if an application
	// manifest specifies use of ComCtl32.dll version 6 or later to enable
	// visual styles.  Otherwise, any window creation will fail.
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	// Set this to include all the common control classes you want to use
	// in your application.
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinApp::InitInstance();


	// Initialize OLE libraries
	if (!AfxOleInit())
	{
		AfxMessageBox(IDP_OLE_INIT_FAILED);
		return FALSE;
	}

	AfxEnableControlContainer();

	EnableTaskbarInteraction(FALSE);

	// AfxInitRichEdit2() is required to use RichEdit control
	// AfxInitRichEdit2();

	// Standard initialization
	// If you are not using these features and wish to reduce the size
	// of your final executable, you should remove from the following
	// the specific initialization routines you do not need
	// Change the registry key under which our settings are stored
	// TODO: You should modify this string to be something appropriate
	// such as the name of your company or organization
	SetRegistryKey(_T("Local AppWizard-Generated Applications"));


	// To create the main window, this code creates a new frame window
	// object and then sets it as the application's main window object
	CMDIFrameWnd* pFrame = new CMainFrame;
	if (!pFrame)
		return FALSE;
	m_pMainWnd = pFrame;
	// create main MDI frame window
	if (!pFrame->LoadFrame(IDR_MAINFRAME))
		return FALSE;
	// try to load shared MDI menus and accelerator table
	//TODO: add additional member variables and load calls for
	//	additional menu types your application may need
	HINSTANCE hInst = AfxGetResourceHandle();
	m_hMDIMenu = ::LoadMenu(hInst, MAKEINTRESOURCE(IDR_minillmmfcTYPE));
	m_hMDIAccel = ::LoadAccelerators(hInst, MAKEINTRESOURCE(IDR_minillmmfcTYPE));




	// The main window has been initialized, so show and update it
	pFrame->ShowWindow(m_nCmdShow);
	pFrame->UpdateWindow();

	OnFileNew();
	return TRUE;
}

int CMiniLLMApp::ExitInstance()
{
	//TODO: handle additional resources you may have added
	if (m_hMDIMenu != nullptr)
		FreeResource(m_hMDIMenu);
	if (m_hMDIAccel != nullptr)
		FreeResource(m_hMDIAccel);

	AfxOleTerm(FALSE);

	return CWinApp::ExitInstance();
}

// CMiniLLMApp message handlers

void CMiniLLMApp::OnFileNew()
{
	CNetSettingsDlg dlg;
	if (dlg.DoModal() == IDOK)
	{
		CMainFrame* pFrame = STATIC_DOWNCAST(CMainFrame, m_pMainWnd);
		// create a new MDI child window
		if (auto pChild{ static_cast<CChildFrame*>(pFrame->CreateNewChild(
			RUNTIME_CLASS(CChildFrame), IDR_minillmmfcTYPE, m_hMDIMenu, m_hMDIAccel)) })
		{
			pChild->GetView().Initialize(dlg);
		}
	}
}

// CAboutDlg dialog used for App About

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg() noexcept;

	// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

	// Implementation
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() noexcept : CDialogEx(IDD_ABOUTBOX)
{}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()

// App command to run the dialog
void CMiniLLMApp::OnAppAbout()
{
	CAboutDlg aboutDlg;
	aboutDlg.DoModal();
}

// CMiniLLMApp message handlers

// CMiniLLMApp:
// See mini-llm-mfc.cpp for the implementation of this class
//
std::string CStringToUtf8(CString const& s)
{
	if (s.IsEmpty()) return {};
	int const n{ WideCharToMultiByte(CP_UTF8, 0, s.GetString(), s.GetLength(), nullptr, 0, nullptr, nullptr) };
	std::string out((size_t)(n > 0 ? n : 0), '\0');
	if (n > 0) WideCharToMultiByte(CP_UTF8, 0, s.GetString(), s.GetLength(), out.data(), n, nullptr, nullptr);
	return out;
}

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

void CopyListCtrlToClipboard(CListCtrl& list)
{
	auto const columns = list.GetHeaderCtrl()->GetItemCount();
	auto const rows = list.GetItemCount();

	std::wstring csv;

	auto AppendField = [&csv](std::wstring const& field)
		{
			//csv += L'"';

			for (auto ch : field)
			{
				//if (ch == L'"')
				//	csv += L'"';

				csv += ch;
			}

			//csv += L'"';
		};

	// Column headers
	for (int col = 0; col < columns; ++col)
	{
		if (col)
			csv += L'\t';

		wchar_t text[1024]{};
		HDITEM item{};
		item.mask = HDI_TEXT;
		item.pszText = text;
		item.cchTextMax = static_cast<int>(std::size(text));

		list.GetHeaderCtrl()->GetItem(col, &item);

		AppendField(text);
	}

	csv += L"\r\n";

	// Rows
	for (int row = 0; row < rows; ++row)
	{
		for (int col = 0; col < columns; ++col)
		{
			if (col)
				csv += L'\t';

			auto text = list.GetItemText(row, col);
			AppendField(text.GetString());
		}

		csv += L"\r\n";
	}

	// Copy Unicode text to the clipboard
	if (!OpenClipboard(list.GetSafeHwnd()))
		return;

	EmptyClipboard();

	auto const bytes = (csv.size() + 1) * sizeof(wchar_t);
	auto hMem = GlobalAlloc(GMEM_MOVEABLE, bytes);

	if (hMem)
	{
		if (auto pMem = GlobalLock(hMem))
		{
			memcpy(pMem, csv.c_str(), bytes);
			GlobalUnlock(hMem);

			if (!SetClipboardData(CF_UNICODETEXT, hMem))
				GlobalFree(hMem); // Ownership transfers only on success.
		}
		else
		{
			GlobalFree(hMem);
		}
	}

	CloseClipboard();
}
