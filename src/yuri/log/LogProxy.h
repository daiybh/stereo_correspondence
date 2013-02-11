/*
 * LogProxy.h
 *
 *  Created on: 11.2.2013
 *      Author: neneko
 */

#ifndef LOGPROXY_H_
#define LOGPROXY_H_
#include <ostream>
#include <sstream>
#include <boost/thread.hpp>
#if __cplusplus >=201103L
#include <future>
#endif
namespace yuri {
namespace log {
struct guarded_stream {
	void write(const std::string msg) {
		boost::mutex::scoped_lock l(mutex_);
		str_ << msg;
	}
	guarded_stream(std::ostream& str):str_(str) {}
private:
	std::ostream& str_;
	boost::mutex mutex_;
};

class LogProxy {
private:
	typedef std::basic_ostream<char>& (*iomanip_t)(std::basic_ostream<char>&);
public:
	LogProxy(guarded_stream& str_,bool dummy):stream_(str_),dummy_(dummy) {}
	LogProxy(const LogProxy& other):stream_(other.stream_),dummy_(other.dummy_) {
		buffer_.str(other.buffer_.str());
	}
	template<typename T>
	LogProxy& operator<<(const T& val_)
	{
		if (!dummy_) {
			buffer_ << val_;
		}
		return *this;
	}

	LogProxy& operator<<(iomanip_t manip)
	{
		// We can't call endl on stringstream, so let's filter it out
		if (manip==static_cast<iomanip_t>(std::endl)) return *this << "\n";
		else return *this << manip;
	}

	~LogProxy() {
#if __cplusplus >=201103L
		const std::string msg = buffer_.str();
		if (!dummy_) std::async([&msg,this](){stream_.write(msg);});
#else
		if (!dummy_) stream_.write(buffer_.str());
#endif
	}
private:
	guarded_stream& stream_;
	std::stringstream buffer_;
	bool dummy_;
};

}
}


#endif /* LOGPROXY_H_ */
