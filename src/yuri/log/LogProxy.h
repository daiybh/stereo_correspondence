/*!
 * @file 		LogProxy.h
 * @author 		Zdenek Travnicek
 * @date 		11.2.1013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef LOGPROXY_H_
#define LOGPROXY_H_
#include "yuri/core/utils/new_types.h"
#include <ostream>
#include <sstream>

namespace yuri {
namespace log {
/*!
 * @brief 		Wrapper struct for std::basic_ostream providing locking
 */
template<
    class CharT,
    class Traits = std::char_traits<CharT>
>
struct guarded_stream {
	typedef std::basic_string<CharT, Traits> string_t;
	typedef std::basic_ostream<CharT, Traits> stream_t;
	typedef CharT char_t;
	/**
	 * @brief 		Writes a string to the contained ostream
	 * @param msg	String to write
	 */
	void write(const string_t msg) {
		yuri::lock_t l(mutex_);
		str_ << msg;
	}

	template<class T>
	guarded_stream& operator<<(const T& val) {
		yuri::lock_t _(mutex_);
		str_ << val;
		return *this;
	}
	guarded_stream(stream_t& str):str_(str) {}
	~guarded_stream() noexcept {}
	char_t widen(char c) { return str_.widen(c); }
private:
	stream_t& str_;
	yuri::mutex mutex_;
};

/**
 * @brief Proxy for output stream
 *
 * LogProxy wraps an output stream and ensures that lines are written correctly
 * even when logging concurrently from several threads
 */
template<
    class CharT,
    class Traits = std::char_traits<CharT>
>
class LogProxy {
private:
	typedef std::basic_ostream<CharT, Traits>& (*iomanip_t)(std::basic_ostream<CharT, Traits>&);
public:
	typedef guarded_stream<CharT, Traits> gstream_t;
	typedef std::basic_stringstream<CharT, Traits> sstream_t;
	/*!
	 * @param	str_	Reference to a @em guarded_stream to write the messages
	 * @param	dummy	All input is discarded, when dummy is set to true
	 */
	LogProxy(gstream_t& str_,bool dummy):stream_(str_),dummy_(dummy) {}
	
	LogProxy(const LogProxy&) = delete;
	/*!
	* @brief			Move constructor. Invalides the original LogProxy object
	*/

	LogProxy(LogProxy&& other) noexcept
		:stream_(other.stream_), dummy_(std::move(other.dummy_)) {
		if (!dummy_) {
			buffer_ << other.buffer_.str();
			other.dummy_ = true;
		}
	}
#ifndef REF_QUALIFIED_MF_UNSUPPORTED
	/*!
	 * @brief			Provides ostream like operator << for inserting messages
	 * @tparam	T		Type of message to insert
	 * @param	val_	Message to write
	 */
	template<typename T>
	LogProxy& operator<<(const T& val_) &
	{
		if (!dummy_) {
			buffer_ << val_;
		}
		return *this;
	}

	/*!
	 * @brief			Provides ostream like operator << for inserting messages
	 * @tparam	T		Type of message to insert
	 * @param	val_	Message to write
	 */
	template<typename T>
	LogProxy&& operator<<(const T& val_) &&
	{
		if (!dummy_) {
			buffer_ << val_;
		}
		return std::move(*this);
	}
#else
	/*!
	 * @brief			Provides ostream like operator << for inserting messages
	 * @tparam	T		Type of message to insert
	 * @param	val_	Message to write
	 */
	template<typename T>
	LogProxy& operator<<(const T& val_)
	{
		if (!dummy_) {
			buffer_ << val_;
		}
		return *this;
	}
#endif
	/*!
	 * @brief 			Overloaded operator<< for manipulators
	 *
	 * We are storing internally to std::stringstream, which won't accept std::endl,
	 * so this method simply replaces std::endl with newlines.
	 * @param	manip	Manipulator for the stream
	 */
	LogProxy& operator<<(iomanip_t manip)
	{
		if (!dummy_) {
			// We can't call endl on std::stringstream, so let's filter it out
			if (manip==static_cast<iomanip_t>(std::endl)) return *this << stream_.widen('\n');
			else return *this << manip;
		}
		return *this;
	}

	~LogProxy() noexcept {
		if (!dummy_) {
			/*const typename gstream_t::string_t str = buffer_.str();
			if (str.size()>0 && str[str.size()-1]!=stream_.widen('\n')) */ buffer_<<stream_.widen('\n');
			// Avoiding unnecessary copy of the rdbuf by writing it directly
			stream_ << buffer_.rdbuf();
		}
	}
private:
	gstream_t& stream_;
	sstream_t buffer_;
	bool dummy_;
};

}
}


#endif /* LOGPROXY_H_ */
