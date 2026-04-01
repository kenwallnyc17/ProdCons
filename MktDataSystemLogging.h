#ifndef MKTDATASYSTEMLOGGING_H_INCLUDED
#define MKTDATASYSTEMLOGGING_H_INCLUDED

#include "MktDataSystemContainers.h"

namespace MKTDATASYSTEM::LOGGING
{
/// !! ordering is important !!
enum class Level {TRACE,DEBUG,INFO,STATS,ERR,WARN,NONE};

const char* tochar(Level lvl)
{
    switch(lvl)
    {
        using enum Level;
    case NONE:
        return "NONE";
    case INFO:
        return "INFO";
    case STATS:
        return "STATS";
    case TRACE:
        return "TRACE";
    case ERR:
        return "ERROR";
    case WARN:
        return "WARN";
    case DEBUG:
        return "DEBUG";

    default:
        return "UNKNOWN";
    }
}

std::ostream& operator<<(std::ostream& os, Level lvl)
{
    return (os << tochar(lvl));
}

using ThreadIDType = std::thread::id;

struct LogEntryHdr
{
    enum class EntryType{LOGENTRY,STOP};
    static constexpr size_t THRD_NAME_MAX = 16;

    ThreadIDType    threadID;
    timespec        tm;
    EntryType       entry_t;
    Level           lvl;
    char            thrdName[THRD_NAME_MAX];
    const char*     fileName;
    int             line;
};

static_assert(sizeof(LogEntryHdr) == 64);

struct LogStreamBuf : std::streambuf
{
    static constexpr size_t INIT_SZ{1<<15};
    static constexpr size_t GROW_SZ{1<<14};

    static_assert(INIT_SZ > sizeof(LogEntryHdr));

    LogStreamBuf() : charstrm_(INIT_SZ)
    {
        reset();
        LogEntryHdr* lrh = reinterpret_cast<LogEntryHdr*>(&charstrm_[0]);
        lrh->threadID = this_thread::get_id();
        strncpy(lrh->thrdName," Thread 1 ",10);/// tmp
        lrh->entry_t = LogEntryHdr::EntryType::LOGENTRY;
    }

    void reset()
    {
        char* bb = &charstrm_[0] + sizeof(LogEntryHdr);
        char* be = &charstrm_[0] + charstrm_.size();

        setp(bb,be);
    }
    void setInfo(const char* fn, int line, Level lvl)
    {
        LogEntryHdr* lrh = reinterpret_cast<LogEntryHdr*>(&charstrm_[0]);
        lrh->lvl = lvl;
        lrh->fileName = fn;
        lrh->line = line;
    }

private:
    int sync();

    int overflow(int c)
    {
        auto seqOff{charstrm_.size()};
        size_t newSz{seqOff + GROW_SZ};

        charstrm_.resize(newSz);

        char* bb = &charstrm_[0] + seqOff;
        char* be = bb + GROW_SZ;

        setp(bb,be);

        if(c != EOF)
        {
            *pptr() = static_cast<char>(c);
            pbump(1);
        }

        return 1;
    }

    vector<char>    charstrm_;
};

struct ThreadLoggingStreams
{
    struct LoggingStream
    {
        LogStreamBuf* strBuf{};
        std::ostream* logstrm{};
    };
    using StreamMap = map<ThreadIDType, LoggingStream>;

    ~ThreadLoggingStreams()
    {
        while(!strmMap_.empty())
        {
            LoggingStream& logs = strmMap_.begin()->second;
            delete logs.strBuf;
            delete logs.logstrm;
            strmMap_.erase(strmMap_.begin());
        }
    }

    std::ostream* getThreadOStream(const char* fn, int line, Level lvl)
    {
        ThreadIDType self{this_thread::get_id()};

        StreamMap::iterator smit{};

        {
            std::scoped_lock lk(mtx_);

            smit = strmMap_.find(self);

            if(smit == strmMap_.end())
            {
                LoggingStream logs;
                logs.strBuf = new LogStreamBuf;
                logs.logstrm = new std::ostream(logs.strBuf);

                auto ist{strmMap_.insert({self, logs})};
                smit = ist.first;
            }
        }

        smit->second.strBuf->setInfo(fn, line, lvl);

        return smit->second.logstrm;
    }
    void deleteThreadOStream()
    {
        ThreadIDType self{this_thread::get_id()};

        std::scoped_lock lk(mtx_);

        auto smit{strmMap_.find(self)};

        if(smit != strmMap_.end())
        {
            LoggingStream logs{smit->second};
            delete logs.strBuf;
            delete logs.logstrm;
            strmMap_.erase(smit);
        }
    }

private:

    std::mutex mtx_{};
    StreamMap strmMap_;
};

/// use as Singleton
struct Logger
{
private:
    Logger(){}
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

public:
    static Logger& inst()
    {
        static Logger ob;
        return ob;
    }

    ~Logger()
    {
        stop();
        ///???
    }

    /// following few functions are static because they are accessed by the TickProcessor threads
    static Level logLevelThreshold(){return logLevelThreshold_;}
    static std::ostream* logStream(const char* fn, int line, Level lvl){return logStreams_.getThreadOStream(fn, line, lvl);}
    static void deleteLogStream(){logStreams_.deleteThreadOStream();}
    static void wrtLogEntry(char* strbuf, size_t len)
    {
//        cout << "\n to log Queue:" << __FILE__ << " : "<< __LINE__ << " : static void wrtLogEntry(char* strbuf, size_t len) " << len <<endl;

        LogEntryHdr* leh{reinterpret_cast<LogEntryHdr*>(strbuf)};

        /// set lrh->logtm_
        clock_gettime(CLOCK_REALTIME, &leh->tm);

        logQ_.write(strbuf, len);
    }


    int init(const string& lfn, Level lvl, uint64_t flSizeMax = 0)
    {
        int ret{};
        if((ret = setFile(lfn)) != 0) return ret;
        if((ret = setLogLevelThreshold(lvl)) != 0) return ret;

        return setFileSizeMax(flSizeMax);
    }
    int start()
    {
        cout << __FILE__ << " : "<< __LINE__ << " : logging :: static int start() " <<endl;
        if(started_ == true) return -1;

        started_ = true;

        auto loggerThread = [this]()
        {
            uint64_t lbytes{};

            if(logFN_.empty() == false)
            {
                plogFile_ = fopen(logFN_.c_str(), "a");
                if(plogFile_ == NULL)
                {
                    ///log error
                    cout << __FILE__ << " : "<< __LINE__ << " : ERROR open log file" <<endl;
                }
                else
                {
                    lbytes = ftell(plogFile_);

                    if(setvbuf(plogFile_, NULL, _IOLBF, 8192) != 0)
                    {
                        // log error
                        cout << __FILE__ << " : "<< __LINE__ << " : ERROR setvbuf" <<endl;
                    }
                }
            }

            if(plogFile_ == 0){plogFile_ = stdout;}

            /// want to get rid of this
            char readbuf[page_size_bytes];/// KMW
            while(true)
            {
              //  int* lrsz = reinterpret_cast<int*>();
                size_t len{};
                logQ_.read1(readbuf, len);

                LogEntryHdr* lrh = reinterpret_cast<LogEntryHdr*>(readbuf);

//                cout << __FILE__ << " : "<< __LINE__ << " : loggerThread() lrh->entry_t : " << to_underlying(lrh->entry_t) <<endl;

                if(lrh->entry_t == LogEntryHdr::EntryType::LOGENTRY)
                {
                    if(logMaxSz_ > 0 && lbytes >= logMaxSz_ && plogFile_ != stdout)
                    {
                        lbytes = 0;
                        swapLogFile(lrh->tm);
                    }

                  //  readbuf[len] = '\0';

                    int lb{log(lrh, len)};
                    if(lb > 0) lbytes += lb;
                }
                else if(lrh->entry_t == LogEntryHdr::EntryType::STOP){break;}

            }

            cout << __FILE__ << " : "<< __LINE__ << " : loggerThread() done" <<endl;
            if(plogFile_ != stdout) fclose(plogFile_);

            return nullptr;
        };

        logThrd_ = move(thread(loggerThread));

        logThId_ = logThrd_.get_id();

        return 1;
    }
    int stop()
    {
        if(started_ == false)return -1;

        started_ = false;

        LogEntryHdr leh;
        leh.entry_t = LogEntryHdr::EntryType::STOP;

        logQ_.write(reinterpret_cast<char*>(&leh), sizeof(LogEntryHdr));

        cout << __FILE__ << " : "<< __LINE__ << " : static int stop() after queue write" <<endl;

        logThrd_.join();

     //   fflush(plogFile_);

        return 17; //pthread_join(logThId_,0);
    }
    int setLogLevelThreshold(const Level lvl)
    {
        logLevelThreshold_ = lvl;

        return 0;
    }

private:
    int setFile(const string& flnm)
    {
        cout << __FILE__ << " : "<< __LINE__ << " : static int setFile(const string& flnm) : " << flnm <<endl;

        std::scoped_lock lk(strmMtx_);

        if(logFN_.empty() == false || started_ == true){return -1;}

        logFN_ = flnm;

        return 0;
    }
    int setFileSizeMax(uint64_t mx)
    {
        if(logMaxSz_ > 0 || started_ == true) return -1;
cout << __FILE__ << " : "<< __LINE__ << " : static int setFileSizeMax(uint64_t mx) : " << mx <<endl;
        logMaxSz_ = mx;

        return 0;
    }
    int log(LogEntryHdr* leh, int len)
    {
 //       cout << "\n to log file: " << __FILE__ << " : "<< __LINE__ << " : static int log(LogEntryHdr* lrh, int len) : " << leh->fileName << ", " << len <<endl;
        const char* tNm{leh->thrdName};

        char* data{reinterpret_cast<char*>(leh+1)};
        timespec* tms{&leh->tm};

        tm* ltm = localtime(&leh->tm.tv_sec);

        int entryLen = fprintf(plogFile_, "%02d/%02d/%04d %02d:%02d:%02d.%06d %5s %-16s: %s",
                              ltm->tm_mon + 1,
                              ltm->tm_mday,
                              ltm->tm_year+1900,
                              ltm->tm_hour,
                              ltm->tm_min,
                              ltm->tm_sec,
                              (int)tms->tv_nsec/1000,
                              tochar(leh->lvl),
                              tNm,
                              data);

        entryLen += fprintf(plogFile_, " [%s:%d]\n", /*basename(*/leh->fileName, leh->line);

      ///  fflush(plogFile_);

        return entryLen;
    }
    void swapLogFile(const timespec& logTm)
    {
        time_t logSec{logTm.tv_sec};

        tm* ltm = localtime(&logSec);
        char dtm[32];
        sprintf(dtm, "%d%02d%02d.%02d%02d%02d",
                ltm->tm_year+1900,
                ltm->tm_mon+1,
                ltm->tm_mday,
                ltm->tm_hour,
                ltm->tm_min,
                ltm->tm_sec);

        string newFNm{logFN_ + "." +dtm};

        cout << __FILE__ << " : "<< __LINE__ << " : swapLogFile : " << newFNm <<endl;

        struct stat stb;
        if(stat(newFNm.c_str(), &stb) == 0)//new file exists
        {
            logMaxSz_ = 0;
            fprintf(plogFile_,"cant rename %s to %s: %s already exists\n", logFN_.c_str(), newFNm.c_str(), newFNm.c_str());
        }
        else if(rename(logFN_.c_str(), newFNm.c_str()) != 0)
        {
            logMaxSz_ = 0;
            fprintf(plogFile_,"renaming %s to %s failed: %d: %s\n", logFN_.c_str(), newFNm.c_str(), errno, strerror(errno));
        }

        fclose(plogFile_);

        plogFile_ = fopen(logFN_.c_str(), "a");

        if(plogFile_ == NULL)
        {
            ///log error
            cout << __FILE__ << " : "<< __LINE__ << " : ERROR in swapLogFile" <<endl;
        }
        else
        {
            if(setvbuf(plogFile_, NULL, _IOLBF, 8192) != 0)
            {
                    // log stderror
                cout << __FILE__ << " : "<< __LINE__ << " : ERROR setvbuf in swapLogFile" <<endl;
            }
        }
    }

    /// these are accessed by the TickProcessor trheads
    inline static ThreadLoggingStreams logStreams_{};
    inline static MKTDATASYSTEM::CONTAINERS::MtoNVariable_2025 logQ_ = MKTDATASYSTEM::CONTAINERS::MtoNVariable_2025(page_size_bytes);
    inline static Level logLevelThreshold_{Level::INFO};
    inline static std::mutex strmMtx_{};

    std::thread logThrd_{};
    ThreadIDType logThId_{};
    std::FILE *plogFile_{};
    string logFN_{};
    uint64_t logMaxSz_{};
    volatile bool started_{};
};

int LogStreamBuf::sync()
{
    *pptr() = '\0';
    size_t dataLen = pptr() - &charstrm_[0];

//    cout << __FILE__ << " : "<< __LINE__ << " : int LogStreamBuf::sync() : " << dataLen <<endl;

    Logger::wrtLogEntry(&charstrm_[0], dataLen+1);/// +1 is for adding a '\0' to end of write
    reset();

    return 1;
}

struct SRLog
{
    ~SRLog(){ /*cout << "\n ~SRLog() \n" << endl;*/ (*logStr_) << std::flush;}
    SRLog(const char* fn, int line, Level lvl) : logStr_(Logger::logStream(fn, line, lvl)){}
/**
    template<IsLoggableContainer OutT>
    SRLog& operator<<(const OutT& outp);

    template<IsNotLoggableContainer OutT>
    SRLog& operator<<(const OutT& outp);

    template<IsLoggableMap OutT>
    SRLog& operator<<(const OutT& outp);
*/
    template<typename U>
    SRLog& operator<<(const U&);

  //  template<typename U>
  //  SRLog& operator<<(U&);

private:
    std::ostream* logStr_;
};

template<typename U>
SRLog& SRLog::operator<<(const U& output)
{
    (*logStr_) << output;

    return *this;
}
/**
struct ErrnoInfoLogMan
{

};

extern const ErrnoInfoLogMan errno_info;

template<>
SRLog& SRLog::operator<< <ErrnoInfoLogMan>(const ErrnoInfoLogMan& manip)
{
    char reason[ErrnoInfoLogMan];
    (*logStr_) << "errno " << errno <<": " << get_errno_msg(errno, reason);

    return *this;
}
*/

struct BufferLogManip
{
    BufferLogManip(const char* buf, size_t sz) : buffer_(buf), sz_(sz){}

    const char* buffer_;
    size_t sz_;
};

inline BufferLogManip buffer(const char* buf, size_t sz) {return BufferLogManip(buf,sz);}

template<>
SRLog& SRLog::operator<< <BufferLogManip>(const BufferLogManip& manip)
{
    for(size_t i = 0; i < manip.sz_; ++i)
    {
        if(std::isprint(manip.buffer_[i]))
        {
            (*logStr_) << manip.buffer_[i];
        }
        else
        {
            (*logStr_) << "*";
        }

        (*logStr_) << "(" << i << "|" << static_cast<int32_t>(manip.buffer_[i]) << ") ";
    }

    return *this;
}


}

using namespace MKTDATASYSTEM::LOGGING;

#define SRLOG(LVL) \
    if(LVL >= Logger::logLevelThreshold()) \
        SRLog(__FILE__,__LINE__,LVL)


#endif // MKTDATASYSTEMLOGGING_H_INCLUDED
