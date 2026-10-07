// CWorkerThread.cpp : implementation file
//

#include "pch.h"
#include "mini-llm-mfc.h"
#include "CWorkerThread.h"


// CWorkerThread

IMPLEMENT_DYNCREATE(CWorkerThread, CWinThread)

CWorkerThread::CWorkerThread()
{
}

CWorkerThread::~CWorkerThread()
{
}

BOOL CWorkerThread::InitInstance()
{
	// TODO:  perform and per-thread initialization here
	return TRUE;
}

int CWorkerThread::ExitInstance()
{
	// TODO:  perform any per-thread cleanup here
	return CWinThread::ExitInstance();
}

BEGIN_MESSAGE_MAP(CWorkerThread, CWinThread)
END_MESSAGE_MAP()


// CWorkerThread message handlers
