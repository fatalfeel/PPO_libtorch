#ifndef WX_LOCKER_H
#define WX_LOCKER_H

class wxCriticalSection
{
public:
#if defined(_WIN32)
	typedef char wxCritSectBuffer[24];
#endif

	// ctor & dtor
    wxCriticalSection()
	{
#if defined(_WIN32)
    	::InitializeCriticalSection((CRITICAL_SECTION *)m_buffer);
#else
    	//pthread_mutex_init(&m_mutex, NULL);
		pthread_mutexattr_t	attr;
    	pthread_mutexattr_init(&attr);
    	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    	pthread_mutex_init(&m_mutex, &attr);
    	pthread_mutexattr_destroy(&attr);
#endif

	}
	
	~wxCriticalSection()
	{
#if defined(_WIN32)
		::DeleteCriticalSection((CRITICAL_SECTION *)m_buffer);
#else
		pthread_mutex_destroy(&m_mutex);
#endif
	}

    // enter the section (the same as locking a mutex)
    void Enter()
	{
#if defined(_WIN32)
    	::EnterCriticalSection((CRITICAL_SECTION *)m_buffer);
#else
    	pthread_mutex_lock(&m_mutex);
#endif
	}

    // leave the critical section (same as unlocking a mutex)
    void Leave()
	{
#if defined(_WIN32)
    	::LeaveCriticalSection((CRITICAL_SECTION *)m_buffer);
#else
    	pthread_mutex_unlock(&m_mutex);
#endif
	}

private:
#if defined(_WIN32)
    wxCritSectBuffer 	m_buffer;
#else
	pthread_mutex_t		m_mutex;
#endif
	
    wxCriticalSection(const wxCriticalSection&); 
	wxCriticalSection& operator=(const wxCriticalSection&);
};

class wxCriticalLocker
{
public:
    wxCriticalLocker(wxCriticalSection& cs) : m_critsect(cs)
    {
        m_critsect.Enter();
    }

    ~wxCriticalLocker()
    {
        m_critsect.Leave();
    }

private:
    wxCriticalSection& m_critsect;

	wxCriticalLocker(const wxCriticalLocker&);           
	wxCriticalLocker& operator=(const wxCriticalLocker&);
};

#endif

