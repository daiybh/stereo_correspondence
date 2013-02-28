/*!
 * @file 		BasicPipe.cpp
 * @author 		Zdenek Travnicek
 * @date 		28.7.2010
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2010 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "BasicPipe.h"
#include <boost/algorithm/string.hpp>
#include <boost/assign/list_of.hpp>
#include <boost/foreach.hpp>
namespace yuri {

FormatInfo_t FormatInfo::raw_format( std::string name, std::vector<std::string> shortnames,
			bool compressed, size_t bpp, std::vector<yuri::size_t> component_depths,
			std::vector<std::string> components, std::vector<yuri::size_t> xsub,
			std::vector<yuri::size_t> ysub)
{
	FormatInfo_t f(new struct FormatInfo(YURI_TYPE_VIDEO, name, shortnames,
			compressed, bpp, component_depths, components, xsub, ysub));
	return f;
}
FormatInfo_t FormatInfo::image_format( std::string name, std::vector<std::string> shortnames,
			std::vector<std::string> mime_types)
{
	FormatInfo_t f(new struct FormatInfo(YURI_TYPE_IMAGE, name, shortnames, true));
	f->mime_types = mime_types;
	return f;
}
FormatInfo_t FormatInfo::video_format( std::string name, std::vector<std::string> shortnames,
			std::vector<std::string> mime_types)
{
	FormatInfo_t f(new struct FormatInfo(YURI_TYPE_VIDEO,name, shortnames, true));
	f->mime_types = mime_types;
	return f;
}

namespace core {

using boost::iequals;
using boost::assign::list_of;
using boost::assign::map_list_of;
pBasicPipe BasicPipe::generator(log::Log &log,std::string name, Parameters &parameters)
{
	shared_ptr<BasicPipe> p (new BasicPipe(log,name));
std::string spolicy = parameters["limit_type"].get<std::string>();
	if (iequals(spolicy,"size")) {
		p->set_limit(parameters["limit"].get<yuri::size_t>(),YURI_DROP_SIZE);
	} else if (iequals(spolicy,"count")) {
		p->set_limit(parameters["limit"].get<yuri::size_t>(),YURI_DROP_COUNT);
	}
	return p;
}
pParameters BasicPipe::configure()
{
	pParameters p (new Parameters());
	(*p)["limit"]=0;
	(*p)["limit_type"]=std::string("none");
	return p;
}

boost::mutex BasicPipe::format_lock;
std::map<yuri::format_t, const FormatInfo_t> BasicPipe::formats=map_list_of<yuri::format_t, FormatInfo_t>
		(YURI_FMT_NONE, 		FormatInfo::raw_format("None", list_of<std::string>("none"), false, 0))
		////////////////// Single component formats //////////////////
		(YURI_FMT_RED8, 		FormatInfo::raw_format("Red 8bit", list_of<std::string>("red8")("red")("r")("r8"), false, 8,list_of(8),list_of("R"),list_of(1),list_of(1)))
		(YURI_FMT_GREEN8, 		FormatInfo::raw_format("Green 8bit", list_of<std::string>("green8")("green")("g8")("g"), false, 8,list_of(8),list_of("G"),list_of(1),list_of(1)))
		(YURI_FMT_BLUE8, 		FormatInfo::raw_format("Blue 8bit", list_of<std::string>("blue8")("blue")("b")("b8"), false, 8,list_of(8),list_of("B"),list_of(1),list_of(1)))
		(YURI_FMT_Y8, 			FormatInfo::raw_format("Luminance 8bit", list_of<std::string>("Y8")("y")("grey8")("grey"), false, 8,list_of(8),list_of("Y"),list_of(1),list_of(1)))
		(YURI_FMT_U8, 			FormatInfo::raw_format("Chroma (U) 8bit", list_of<std::string>("U8")("u"), false, 8,list_of(8),list_of("U"),list_of(1),list_of(1)))
		(YURI_FMT_V8,	 		FormatInfo::raw_format("Chroma (V) 8bit", list_of<std::string>("V8")("v"), false, 8,list_of(8),list_of("V"),list_of(1),list_of(1)))
		(YURI_FMT_DEPTH8, 		FormatInfo::raw_format("Depth 8bit", list_of<std::string>("DEPTH8")("depth")("d8"), false, 8,list_of(8),list_of("D"),list_of(1),list_of(1)))
		(YURI_FMT_RED16, 		FormatInfo::raw_format("Red 16bit", list_of<std::string>("red16")("r16"), false, 16,list_of(16),list_of("R"),list_of(1),list_of(1)))
		(YURI_FMT_GREEN16, 		FormatInfo::raw_format("Green 16bit", list_of<std::string>("green16")("g16"), false, 16,list_of(16),list_of("G"),list_of(1),list_of(1)))
		(YURI_FMT_BLUE16, 		FormatInfo::raw_format("Blue 16bit", list_of<std::string>("blue16")("b16"), false, 16,list_of(16),list_of("B"),list_of(1),list_of(1)))
		(YURI_FMT_Y16, 			FormatInfo::raw_format("Luminance 16bit", list_of<std::string>("y16")("grey16"), false, 16,list_of(16),list_of("Y"),list_of(1),list_of(1)))
		(YURI_FMT_U16, 			FormatInfo::raw_format("Chroma (U) 16bit", list_of<std::string>("u16"), false, 16,list_of(16),list_of("U"),list_of(1),list_of(1)))
		(YURI_FMT_V16,	 		FormatInfo::raw_format("Chroma (V) 16bit", list_of<std::string>("v16"), false, 16,list_of(16),list_of("V"),list_of(1),list_of(1)))
		(YURI_FMT_DEPTH16, 		FormatInfo::raw_format("Depth 16bit", list_of<std::string>("DEPTH16")("d16"), false, 16,list_of(16),list_of("D"),list_of(1),list_of(1)))
		(YURI_FMT_SINGLE_COMPONENT16, 		FormatInfo::raw_format("Single 16bit", list_of<std::string>("SINGLE16")("s16"), false, 16,list_of(16),list_of("D"),list_of(1),list_of(1)))
		//////////////////      RGB formats        //////////////////
		(YURI_FMT_RGB, 			FormatInfo::raw_format("RGB 24bit", list_of<std::string>("RGB")("RGB24"), false, 24,list_of(8)(8)(8),list_of("RGB"),list_of(1),list_of(1)))
		(YURI_FMT_RGBA, 		FormatInfo::raw_format("RGBA 32bit", list_of<std::string>("RGBA")("RGB32")("RGBA32"), false, 32,list_of(8)(8)(8)(8),list_of("RGBA"),list_of(1),list_of(1)))
		(YURI_FMT_RGB_PLANAR, 	FormatInfo::raw_format("Planar RGB 24bit", list_of<std::string>("RGBp")("RGB24p"), false, 24,list_of(8)(8)(8),list_of("R")("G")("B"),list_of(1)(1)(1),list_of(1)(1)(1)))
		(YURI_FMT_RGBA_PLANAR, 	FormatInfo::raw_format("Planar RGBA 32bit", list_of<std::string>("RGBAp")("RGB32p")("RGBA32p"), false, 32,list_of(8)(8)(8)(8),list_of("R")("G")("B")("A"),list_of(1)(1)(1)(1),list_of(1)(1)(1)(1)))
		(YURI_FMT_BGR, 			FormatInfo::raw_format("BGR 24bit", list_of<std::string>("BGR")("BGR24"), false, 24,list_of(8)(8)(8),list_of("BGR"),list_of(1),list_of(1)))
		(YURI_FMT_BGRA, 		FormatInfo::raw_format("BGRA 32bit", list_of<std::string>("BGRA")("BGR32")("BGRA32"), false, 32,list_of(8)(8)(8)(8),list_of("BGRA"),list_of(1),list_of(1)))

		//////////////////      YUVformats        //////////////////
		(YURI_FMT_YUV411, 		FormatInfo::raw_format("YUV (YUYV) 4:1:1 12bit", list_of<std::string>("YUV411"), false, 12, list_of(8)(8)(8),list_of("UYYYVYYY"),list_of(1),list_of(1)))
		(YURI_FMT_YUV422, 		FormatInfo::raw_format("YUV (YUYV) 4:2:2 16bit", list_of<std::string>("YUV422")("YUYV"), false, 16, list_of(8)(8)(8),list_of("YUYV"),list_of(1),list_of(1)))
		(YURI_FMT_YUV444, 		FormatInfo::raw_format("YUV (YUYV) 4:4:4 24bit", list_of<std::string>("YUV444"), false, 24, list_of(8)(8)(8),list_of("YUVYUV"),list_of(1),list_of(1)))
		(YURI_FMT_YUV420_PLANAR,FormatInfo::raw_format("YUV Planar (YUYV) 4:2:0 12bit", list_of<std::string>("YUV420P"), false, 12, list_of(8)(8)(8),list_of("Y")("U")("V"),list_of(1)(2)(2),list_of(1)(2)(2)))
		(YURI_FMT_YUV422_PLANAR,FormatInfo::raw_format("YUV Planar (YUYV) 4:2:2 16bit", list_of<std::string>("YUV422P"), false, 16, list_of(8)(8)(8),list_of("Y")("U")("V"),list_of(1)(2)(2),list_of(1)(1)(1)))
		(YURI_FMT_YUV444_PLANAR,FormatInfo::raw_format("YUV Planar (YUYV) 4:4:4 24bit", list_of<std::string>("YUV444P"), false, 16, list_of(8)(8)(8),list_of("Y")("U")("V"),list_of(1)(1)(1),list_of(1)(1)(1)))
		//////////////////      10bit formats        //////////////////
		(YURI_FMT_V210, 		FormatInfo::raw_format("V210 20bit", list_of<std::string>("v210"), false, 20, list_of(10)(10)(10),list_of("YUYV"),list_of(1),list_of(1)))
		(YURI_FMT_R210, 		FormatInfo::raw_format("R210 20bit", list_of<std::string>("r210"), false, 32, list_of(10)(10)(10),list_of("RGB"),list_of(1),list_of(1)))
		////////////////// Compressed raw formats (no header) //////////////////
		(YURI_FMT_DXT1, 		FormatInfo::raw_format("DXT1", list_of<std::string>("dxt1")("bc1"), true, 4))
		(YURI_FMT_DXT2, 		FormatInfo::raw_format("DXT2", list_of<std::string>("dxt2")("bc2"), true, 8))
		(YURI_FMT_DXT3, 		FormatInfo::raw_format("DXT3", list_of<std::string>("dxt3"), true, 8))
		(YURI_FMT_DXT4, 		FormatInfo::raw_format("DXT4", list_of<std::string>("dxt4")("bc3"), true, 8))
		(YURI_FMT_DXT5, 		FormatInfo::raw_format("DXT5", list_of<std::string>("dxt5"), true, 8))

		////////////////// Compressed raw formats with mipmaps (no header) //////////////////

		(YURI_FMT_DXT1_WITH_MIPMAPS,FormatInfo::raw_format("DXT1 with mipmaps", list_of<std::string>("dxt1t"), true, 4))
		(YURI_FMT_DXT2_WITH_MIPMAPS,FormatInfo::raw_format("DXT2 with mipmaps", list_of<std::string>("dxt2t"), true, 8))
		(YURI_FMT_DXT3_WITH_MIPMAPS,FormatInfo::raw_format("DXT3 with mipmaps", list_of<std::string>("dxt3t"), true, 8))
		(YURI_FMT_DXT4_WITH_MIPMAPS,FormatInfo::raw_format("DXT4 with mipmaps", list_of<std::string>("dxt4t"), true, 8))
		(YURI_FMT_DXT5_WITH_MIPMAPS,FormatInfo::raw_format("DXT5 with mipmaps", list_of<std::string>("dxt5t"), true, 8))

		////////////////// Custom formats //////////////////
		(YURI_FMT_V210_MVTP, 	FormatInfo::raw_format("V210 20bit (MVTP ordering)", list_of<std::string>("mvtp_v210")("v210mvtp"), true, 20, list_of(10)(10)(10)))
		(YURI_FMT_MVTP_FULL_FRAME,	FormatInfo::raw_format("MVTP Fullframe", list_of<std::string>("mvtp_full"), true, 20, list_of(10)(10)(10)))
		(YURI_FMT_MVTP_AUX_DATA,FormatInfo::raw_format("MVTP Auxdata", list_of<std::string>("mvtp_aux"), true, 20, list_of(10)(10)(10)))
		(YURI_FMT_BAYER_RGGB,FormatInfo::raw_format("Bayer patter RGGB", list_of<std::string>("bayer")("bayer_rggb"), true, 8))
		////////////////// Image formats  (complete files with headers) //////////////////
		(YURI_IMAGE_JPEG,		FormatInfo::image_format("JPEG",list_of<std::string>("jpeg")("jpg"),list_of<std::string>("image/jpeg")))
		(YURI_IMAGE_PNG,		FormatInfo::image_format("PNG",list_of<std::string>("png"),list_of<std::string>("image/png")))
		(YURI_IMAGE_GIF,		FormatInfo::image_format("GIF",list_of<std::string>("gif"),list_of<std::string>("image/gif")))
		(YURI_IMAGE_TIFF,		FormatInfo::image_format("TIFF",list_of<std::string>("tiff"),list_of<std::string>("image/tiff")))
		(YURI_IMAGE_DDS,		FormatInfo::image_format("DDS",list_of<std::string>("dds"),list_of<std::string>("image/x-dds")))
		(YURI_IMAGE_JPEG2K,		FormatInfo::image_format("JPEG 2000",list_of<std::string>("jpeg2k")("jpg2k")("j2k"),list_of<std::string>("image/jp2")))
		////////////////// Video formats //////////////////
		(YURI_VIDEO_MPEG1,		FormatInfo::image_format("MPEG 1",list_of<std::string>("mpg")("mpeg")("mpg1")("mpeg1"),list_of<std::string>("video/mpeg")))
		(YURI_VIDEO_MPEG2,		FormatInfo::image_format("MPEG 2",list_of<std::string>("mpg2")("mpeg2"),list_of<std::string>("video/mpeg2")))
		(YURI_VIDEO_HUFFYUV,	FormatInfo::image_format("HUFFYUV",list_of<std::string>("huffyuv"),list_of<std::string>("video/x-huffyuv")))
		(YURI_VIDEO_DV,			FormatInfo::image_format("DV",list_of<std::string>("dv"),list_of<std::string>("video/x-dv")))
		(YURI_VIDEO_MJPEG,		FormatInfo::image_format("Motion JPEG",list_of<std::string>("mjpg")("mjpeg"),list_of<std::string>("video/x-motion-jpeg")))
		(YURI_VIDEO_MPEGTS,		FormatInfo::image_format("Mpeg2 Transport Stream",list_of<std::string>("mpg2ts")("mpegts"),list_of<std::string>("video/mp2t")))
		(YURI_VIDEO_H264,		FormatInfo::image_format("H.264",list_of<std::string>("h264")("h.264")("x264"),list_of<std::string>("video/h264")))
		(YURI_VIDEO_FLASH,		FormatInfo::image_format("Flash video",list_of<std::string>("flash"),list_of<std::string>()))
		(YURI_VIDEO_DIRAC,		FormatInfo::image_format("Dirac",list_of<std::string>("dirac"),list_of<std::string>()))
		(YURI_VIDEO_H263,		FormatInfo::image_format("H.263",list_of<std::string>("h263")("h.263"),list_of<std::string>("video/h263")))
		(YURI_VIDEO_H263PLUS,	FormatInfo::image_format("H.263+",list_of<std::string>("h263+")("h263plus")("h.263+"),list_of<std::string>("video/h263p")))
		(YURI_VIDEO_THEORA,		FormatInfo::image_format("Theora",list_of<std::string>("theora"),list_of<std::string>("video/theora")))
		(YURI_VIDEO_VP8,		FormatInfo::image_format("VP8",list_of<std::string>("vp8"),list_of<std::string>("video/vp8")))


;
/*
std::map<std::string,yuri::format_t,compare_insensitive> BasicPipe::mime_to_format;
std::map<yuri::format_tstd::string> BasicPipe::format_to_mime;
mutex BasicPipe::mime_conv_mutex;
*/


BasicPipe::BasicPipe(log::Log &log_, std::string name):log(log_),name(name),
	type(YURI_TYPE_NONE),discrete(true),changed(true),
	notificationsEnabled(true),closed(false),bytes(0),count(0),limit(0),
	totalBytes(0),totalCount(0),dropped(0),dropPolicy(YURI_DROP_NONE)
{
	log.set_label("[Pipe "+name+"] ");
}
/*
 * TODO: statistics won't work on 32bit systems. It might need to be looked at.
 */
BasicPipe::~BasicPipe()
{
std::string units = "Bytes";
	yuri::size_t pbytes = totalBytes;
	if (totalBytes> 1024) {
		if (totalBytes < 1048576) { units = "kB"; pbytes=totalBytes/1024; }
		else if (totalBytes < 1048576L*1024) { units = "MB"; pbytes=totalBytes/1048576; }
		else if (totalBytes/1048576 <  1048576L) { units = "GB"; pbytes=totalBytes/1048576/1024; }
		else { units = "TB"; pbytes=totalBytes/1048576/1048576; }
	}
	log[log::info] << "Processed total of " << totalCount << " frames with size of "
			<< pbytes << " " << units << ". " << dropped
			<< " frames were dropped" << "\n";
}

/*void BasicPipe::push_frame(BasicFrame *frame)
{
	mutex::scoped_lock l(framesLock);
	pBasicFrame p(frame);
	do_push_frame(p);
}*/

void BasicPipe::push_frame(pBasicFrame frame)
{
	boost::mutex::scoped_lock l(framesLock);
	do_push_frame(frame);
}

pBasicFrame BasicPipe::pop_frame()
{
	boost::mutex::scoped_lock l(framesLock);
	return do_pop_frame();
}

pBasicFrame BasicPipe::pop_latest()
{
	boost::mutex::scoped_lock l(framesLock);
	return do_pop_latest();
}

void BasicPipe::set_limit(yuri::size_t limit0, yuri::ubyte_t policy)
{
	mutex::scoped_lock l(framesLock);
	do_set_limit(limit0,policy);
}

void BasicPipe::set_type(yuri::format_t type)
{
	mutex::scoped_lock l(framesLock);
	do_set_type(type);
}

yuri::format_t BasicPipe::get_type()
{
	mutex::scoped_lock l(framesLock);
	return do_get_type();
}

bool BasicPipe::is_changed()
{
	mutex::scoped_lock l(framesLock);
	if (changed) {
		do_set_changed(false);
		return true;
	}
	return false;

}

bool BasicPipe::is_empty()
{
	mutex::scoped_lock l(framesLock);
	return frames.empty();
}

int BasicPipe::get_notification_fd()
{
	mutex::scoped_lock l(framesLock);
	return do_get_notification_fd();
}
void BasicPipe::cancel_notifications()
{
	mutex::scoped_lock l(framesLock);
	do_cancel_notifications();
}
/*
 * 		Protected methods
 */

void BasicPipe::do_set_changed(bool ch)
{
	if ((changed=ch)) {
		do_clear_pipe();
	}
}

void BasicPipe::do_clear_pipe()
{
	while (!frames.empty()) do_pop_frame();
}


void BasicPipe::do_push_frame(pBasicFrame frame)
{
	bytes += frame->get_size();
	count++;
	// Statistics
	totalBytes += frame->get_size();
	totalCount++;
	frames.push(frame);
	switch (dropPolicy) {
		case YURI_DROP_SIZE:
			while (bytes > limit && !frames.empty()) {
				do_pop_frame();
				dropped++;
			}
			assert(bytes <= limit);
			break;
		case YURI_DROP_COUNT:
			while (count > limit && !frames.empty()) {
				do_pop_frame();
				dropped++;
			}
			assert (count <= limit);
			break;
	}
	do_notify();
}


pBasicFrame BasicPipe::do_pop_frame()
{
	if (frames.empty()) {
		return pBasicFrame();
	}
	pBasicFrame frame = frames.front();
	frames.pop();
	assert(bytes>=frame->get_size());
	assert(count>=1);
	bytes-=frame->get_size();
	count--;
	return frame;
}

pBasicFrame BasicPipe::do_pop_latest()
{
	pBasicFrame f0,f;
	while (!frames.empty()) {
		f = do_pop_frame();
		if (f) f0 = f;
	}
	return f0;
}

void BasicPipe::do_set_limit(yuri::size_t l, yuri::ubyte_t policy)
{
	if (l <= 0 || policy == YURI_DROP_NONE) {
		limit = 0;
		dropPolicy = YURI_DROP_NONE;
		log[log::info] << "Removing limits from the pipe" << "\n";
	} else {
		limit = l;
		dropPolicy = policy;
		log[log::info] << "Limiting pipe to " << limit
				<< (policy==YURI_DROP_COUNT?" frames":" bytes") << "\n";
	}
}

void BasicPipe::do_set_type(yuri::format_t type)
{
	if (this->type != type) {
		do_set_changed(true);
		this->type = type;
	}

}
yuri::format_t BasicPipe::do_get_type()
{
	return type;
}

void BasicPipe::close()
{
	mutex::scoped_lock l(framesLock);
	closed=true;
}

bool BasicPipe::is_closed()
{
	mutex::scoped_lock l(framesLock);
	return closed;
}

int BasicPipe::do_get_notification_fd()
{
#ifdef __linux__
	if (!notifySockets.get()) {
		notifySockets.reset(new int [2]);

		socketpair(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK,0,notifySockets.get());
		/*int flags = fcntl(notifySockets[0], F_GETFL, 0);
		fcntl(notifySockets[0], F_SETFL, flags | O_NONBLOCK);
		fcntl(notifySockets[1], F_SETFL, flags | O_NONBLOCK);*/
	}
	notificationsEnabled = true;
	return notifySockets[1];
#else
	notificationsEnabled = false;
	return 0;
#endif

}

void BasicPipe::do_cancel_notifications()
{
	notificationsEnabled = false;
#ifdef __linux__
	::close(notifySockets[0]);
	::close(notifySockets[1]);
#endif
}


void BasicPipe::do_notify()
{
#ifdef __linux__
	if (!notificationsEnabled) return;
	if (!notifySockets.get()) {
		log[log::error] << "No notification sockets!!!!!" << "\n";
		return;
	}
	char c = 0x37;
	int dummy YURI_UNUSED = write(notifySockets[0],&c,1);
#endif
}

/*
 * 		Static methods
 */
std::string BasicPipe::get_type_string(yuri::format_t type)
{
	switch(type) {
		case YURI_TYPE_VIDEO: return std::string("VIDEO");
		case YURI_TYPE_AUDIO: return std::string("AUDIO");
		case YURI_TYPE_TEXT: return std::string("TEXT");

		case YURI_TYPE_NONE:
		default: return std::string("NONE");
	}
}
std::string BasicPipe::get_format_string(yuri::format_t format)
{
	boost::mutex::scoped_lock lock(format_lock);
	if (formats.count(format)) return formats[format]->long_name;
	return std::string("INVALID VALUE");
}

std::string BasicPipe::get_simple_format_string(yuri::format_t format)
{
	boost::mutex::scoped_lock lock(format_lock);
	if (formats.count(format)) {
		if (formats[format]->short_names.size())
			return formats[format]->short_names[0];
	}
	return std::string("NONE");
}

yuri::size_t BasicPipe::get_bpp_from_format(yuri::format_t format)
{
	boost::mutex::scoped_lock lock(format_lock);
	if (formats.count(format)) return formats[format]->bpp;
	return 0;
}

yuri::format_t BasicPipe::get_format_from_string(std::string format, yuri::format_t group)
{
	boost::mutex::scoped_lock lock(format_lock);
	std::pair<yuri::format_t,FormatInfo_t> fmt;
	BOOST_FOREACH(fmt, formats) {
		if (group!=YURI_TYPE_NONE && fmt.second->type !=group) continue;
		BOOST_FOREACH(std::string name, fmt.second->short_names) {
			if (iequals(name,format)) return fmt.first;
		}
	}
	return YURI_FMT_NONE;
}

yuri::format_t BasicPipe::get_format_group(yuri::format_t format)
{
	boost::mutex::scoped_lock lock(format_lock);
	if (formats.count(format)) return formats[format]->type;
	return YURI_FMT_NONE;
}

FormatInfo_t BasicPipe::get_format_info(yuri::format_t format)
{
	boost::mutex::scoped_lock lock(format_lock);
	if (formats.count(format)) return formats[format];
	return FormatInfo_t();
}
/// Tries to guess frame type from the mime type
/// \param frame input frame
/// \param mime mime type of the data
/// \return Returns the type of data (YURI_TYPE_*)
yuri::format_t BasicPipe::set_frame_from_mime(pBasicFrame frame,std::string mime)
{
//	set_mime_types();
//	if (mime_to_format.find(mime)!=mime_to_format.end()) {
//		frame->set_parameters(mime_to_format[mime],0,0);
//		/* We can safely assume the type is video
//		 * as all defined mime-types are image/video.
//		 * Once we have more mime-types, this should be done more cleanly
//		 */
//		return YURI_TYPE_VIDEO;
//	}
	boost::mutex::scoped_lock lock(format_lock);
	std::pair<yuri::format_t,FormatInfo_t> fmt;
	BOOST_FOREACH(fmt, formats) {
		BOOST_FOREACH(std::string m, fmt.second->mime_types) {
			if (iequals(mime,m)) {
				frame->set_parameters(fmt.first,0,0);
				return fmt.second->type;
			}
		}
	}
	return YURI_TYPE_NONE;
}

/*
void BasicPipe::set_mime_types()
{
	mutex::scoped_lock l(mime_conv_mutex);
	if (!mime_to_format.empty()) return;
	add_to_formats("image/jpeg",YURI_IMAGE_JPEG);
	add_to_formats("image/png",YURI_IMAGE_PNG);
	add_to_formats("image/gif",YURI_IMAGE_GIF);
	add_to_formats("image/tiff",YURI_IMAGE_TIFF);
	add_to_formats("video/DV",YURI_VIDEO_DV);
	add_to_formats("video/x-mpegts",YURI_VIDEO_MPEGTS);
}
void BasicPipe::add_to_formatsstd::string mime, yuri::format_t format)
{
	mime_to_format[mime]=format;
	format_to_mime[format]=mime;
}

*/
}

}

