/*!
 * @file 		FileDump.cpp
 * @author 		Zdenek Travnicek
 * @date 		24.7.2010
 * @date		04.2.2014
 * @copyright	Institute of Intermedia, CTU in Prague, 2010 - 2014
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "FileDump.h"
#include <sstream>
#include <iomanip>
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
#include "yuri/core/frame/RawAudioFrame.h"

#ifdef HAVE_BOOST_REGEX
#include "yuri/core/utils/hostname.h"
#include "yuri/version.h"
#include "yuri/core/utils/frame_info.h"
#include <boost/regex.hpp>
#endif
namespace yuri
{
namespace dump
{

MODULE_REGISTRATION_BEGIN("filedump")
	REGISTER_IOTHREAD("filedump",FileDump)
MODULE_REGISTRATION_END()

IOTHREAD_GENERATOR(FileDump)

core::Parameters FileDump::configure()
{
	core::Parameters p = IOFilter::configure();
	p.set_description("Outputs frames to a file. It either dump all frames to one file of every frame to separate file. Filename can contain a format specifiers in a form %n, %t, %T or %05s");
	p["sequence"]["Number of digits in sequence number. Set to 0 to disable sequence and output all frames to one file."]=0;
	p["filename"]["Required parameter. Path of file to dump to"]=std::string();
	p["frame_limit"]["Maximal number of frames to dump. 0 for unlimited"]=0;
	return p;
}

namespace {

template<class Stream, class Value>
Stream& print_formated_value(Stream& os, const Value& value, int width = 0, bool zero = true)
{
	if (zero) {
		os << std::setfill('0');
	}
	if (width > 0) {
		os << std::setw(width);
	}
	os << value;
	return os;
}

template<class T>
std::string append_to_filename(const std::string& filename, const T& value, int width = 0)
{
	std::stringstream ss;
	const auto dot_index = filename.find_last_of('.');
	if (dot_index != std::string::npos) {
		ss << filename.substr(0,dot_index);
	} else {
		ss << filename;
	}
	print_formated_value(ss, value, width);
	if (dot_index != std::string::npos) {
		ss << filename.substr(dot_index);
	}
	return ss.str();
}

#ifdef HAVE_BOOST_REGEX

const boost::regex specifier_pattern ("%(0?\\d+[s]|[ntfFTHDSOv%])");

template<class S1, class S2>
std::string to_s(const std::pair<S1, S2>& p)
{
	return std::string(p.first, p.second);
}

/**
 *
 * @param fname
 * @return pair of two boolean values. First specifies, if there are any specifiers at all,
 *  		second is true, when any of the specifiers changes with the frames.
 */
std::pair<bool,bool> contains_specifiers(const std::string& fname)
{
	bool any_spec = false;
	bool seq_spec = false;
	auto beg = fname.cbegin();
	auto end = fname.cend();

	boost::smatch what;

	while(boost::regex_search(beg, end, what, specifier_pattern, boost::match_default)) {
		any_spec = true;

		assert (std::distance(what[0].first, what[0].second) > 0);
		auto specifier = *(what[0].second - 1);
		switch (specifier) {
			case 't':
			case 'T':
			case 's':
			case 'f':
			case 'F':
				seq_spec = true;
				break;
			default:
				break;
		}

		if (seq_spec) break;

		beg = what[0].second;
	}
	return std::make_pair(any_spec, seq_spec);

}

template<class Value>
std::string parse_and_replace(const std::string& spec, const Value& value)
{
	boost::regex pat ("%(0?)(\\d+)([[:alpha:]])");
	boost::smatch what;
	std::stringstream ss;
	if (boost::regex_match(spec, what, pat)) {
		const bool zero = what[1].first!=what[1].second;
		const auto count = std::stoul(to_s(what[2]));
		print_formated_value(ss, value, count, zero);
	}
	return ss.str();
}


#endif
}



FileDump::FileDump(log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters):
	IOFilter(log_,parent,"Dump"),
	event::BasicEventProducer(log),
	dump_file(),filename(),seq_chars(0),seq_number(0),dumped_frames(0),
	dump_limit(0),use_regex_(false),single_file_(true)
{
	IOTHREAD_INIT(parameters);
	if (filename.empty()) throw exception::InitializationFailed("No filename specified");

#ifdef HAVE_BOOST_REGEX
	auto s = contains_specifiers(filename);
	use_regex_ = s.first;
	single_file_ = !s.second;
	if (single_file_ && seq_chars) {
		std::string spec = "%0"+std::to_string(seq_chars)+"s";
		filename = append_to_filename(filename, spec);
		log[log::info] << "Expanded filename to " << filename;
		use_regex_ = true;
		single_file_ = false;
	}
#else
	single_file_ = seq_number <= 0;
#endif

	if (single_file_)
		dump_file.open(filename.c_str(), std::ios::binary | std::ios::out);
}

FileDump::~FileDump() noexcept
{
	if (dump_file.is_open()) dump_file.close();
}

std::string FileDump::generate_filename(const core::pFrame& frame)
{
	if (!use_regex_ && !single_file_) {
		return append_to_filename(filename, seq_number++, seq_chars);
	}
#ifdef HAVE_BOOST_REGEX
	else {
		std::string new_name;
		auto beg = filename.cbegin();
		auto end = filename.cend();
		boost::smatch what;
		std::stringstream ss;
		while(boost::regex_search(beg, end, what, specifier_pattern, boost::match_default)) {
			assert (std::distance(what[0].first, what[0].second) > 0);
			if (beg != what[0].first) {
				ss << std::string{beg, what[0].first};
			}
			auto specifier = *(what[0].second - 1);
			switch (specifier) {
				case 's':
					ss << parse_and_replace(to_s(what[0]), seq_number);
					break;
				case 'n':
					ss << get_node_name();
					break;
				case 'T':
					ss << timestamp_t{};
					break;
				case 't':
					ss << frame->get_timestamp();
					break;
				case 'H':
					ss << core::utils::get_hostname();
					break;
				case 'D':
					ss << core::utils::get_domain();
					break;
				case 'S':
					ss << core::utils::get_sysver();
					break;
				case 'O':
					ss << core::utils::get_sysname();
					break;
				case 'v':
					ss << yuri_version;
					break;
				case 'f':
					ss << core::utils::get_frame_type_name(frame->get_format(), true);
					break;
				case 'F':
					ss << core::utils::get_frame_type_name(frame->get_format(), false);
					break;
				case '%':
					ss << "%";
					break;
				default:
					break;
			}
			beg = what[0].second;
		}
		if (beg != end) {
			ss << std::string(beg, end);
		}
		return ss.str();
	}
#endif
	return filename;
}

core::pFrame FileDump::do_simple_single_step(core::pFrame frame)
{
	if (!single_file_) {
		const auto seq_filename = generate_filename(frame);
		++seq_number;
		log[log::debug] << "New filename " << seq_filename;
		dump_file.open(seq_filename.c_str(), std::ios::binary | std::ios::out);
		emit_event("filename",seq_filename);
	}
	bool written = true;
	if (auto f = std::dynamic_pointer_cast<core::RawVideoFrame>(frame)) {
		log[log::debug]<<"Dumping " << f->get_planes_count() << " planes";
		for (yuri::size_t i=0; i<f->get_planes_count();++i) {
			dump_file.write(reinterpret_cast<const char *>(PLANE_RAW_DATA(f,0)),PLANE_SIZE(f,0));
		}
	} else if (auto f2 = std::dynamic_pointer_cast<core::CompressedVideoFrame>(frame)) {
		dump_file.write(reinterpret_cast<const char *>(f2->begin()),f2->size());
	} else if (auto f3 = std::dynamic_pointer_cast<core::RawAudioFrame>(frame)) {
		dump_file.write(reinterpret_cast<const char *>(f3->data()),f3->size());
	} else {
		written=false;
	}
	if (!single_file_) {
		dump_file.close();
		if (written) {
			emit_event("sequence",seq_number);
		}
	}
	if (written) {
		emit_event("frame");
	}

	// The comparison is evaluated FIRST in order to have dumped_frames counted even if dump_limit is zero
	if (++dumped_frames >= dump_limit && dump_limit) {
		log[log::debug] << "Maximal number of frames reached, quitting.";
		request_end();
	}

	return {};
}

bool FileDump::set_param(const core::Parameter &param)
{
	if (assign_parameters(param)
			(filename, "filename")
			(seq_chars, "sequence")
			(dump_limit, "frame_limit"))
		return true;
	return IOFilter::set_param(param);
}

}
}
//End of File

