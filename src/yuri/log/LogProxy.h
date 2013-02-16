/*!
 * @file 		LogProxy.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.1013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#ifndef LOGPROXY_H_
#define LOGPROXY_H_
#include "yuri/io/types.h"
#include <ostream>
#include <sstream>
#include <boost/thread.hpp>

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

class EXPORT LogProxy {
private:
	typedef std::basic_ostream<char>& (*iomanip_t)(std::basic_ostream<char>&);
public:
	LogProxy(guarded_stream& str_,bool dummy):stream_(str_),dummy_(dummy) {}
	LogProxy(const LogProxy& other):stream_(other.stream_),dummy_(other.dummy_) {
		if (!dummy_) {
			buffer_.str(other.buffer_.str());
			const_cast<LogProxy&>(other).dummy_=true;
		}
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
		// We can't call endl onstd::stringstream, so let's filter it out
		if (manip==static_cast<iomanip_t>(std::endl)) return *this << "\n";
		else return *this << manip;
	}

	~LogProxy() {
#if 0 && __cplusplus >=201103L
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
