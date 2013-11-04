/*!
 * @file 		V4l2Source.cpp
 * @author 		Zdenek Travnicek
 * @date 		17.5.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "V4l2Source.h"
//#include <boost/algorithm/string.hpp>
#include "yuri/core/Module.h"
#include <string>
//#include <boost/assign.hpp>

#include <unistd.h>
#include <cstring>
#include "yuri/core/frame/raw_frame_params.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/frame/compressed_frame_types.h"
#include "yuri/core/frame/compressed_frame_params.h"
#include "yuri/core/frame/CompressedVideoFrame.h"
namespace yuri {

namespace io {



IOTHREAD_GENERATOR(V4l2Source)
MODULE_REGISTRATION_BEGIN("v4l2source")
	REGISTER_IOTHREAD("v4l2source",V4l2Source)
MODULE_REGISTRATION_END()

namespace {
	using namespace yuri::core::raw_format;
	using namespace yuri::core::compressed_frame;
	std::map<yuri::format_t, uint32_t> formats_map=
			yuri::map_list_of<yuri::format_t, uint32_t>
			(rgb24,		V4L2_PIX_FMT_RGB24)
			(argb32, 	V4L2_PIX_FMT_RGB32)
			(bgr24, 	V4L2_PIX_FMT_BGR24)
			(bgra32, 	V4L2_PIX_FMT_BGR32)
			(rgb15,		V4L2_PIX_FMT_RGB555)
			(rgb16,		V4L2_PIX_FMT_RGB565)
			(yuyv422, 	V4L2_PIX_FMT_YUYV)
			(yvyu422, 	V4L2_PIX_FMT_YVYU)
			(uyvy422, 	V4L2_PIX_FMT_UYVY)
			(vyuy422, 	V4L2_PIX_FMT_VYUY)
			//(yuv420p, V4L2_PIX_FMT_YUV420)
//			(YURI_VIDEO_DV, V4L2_PIX_FMT_DV)
//			(YURI_VIDEO_MJPEG, V4L2_PIX_FMT_MJPEG)
//			(YURI_IMAGE_JPEG, V4L2_PIX_FMT_JPEG)
			(bayer_bggr,V4L2_PIX_FMT_SBGGR8)
			(bayer_rggb,V4L2_PIX_FMT_SRGGB8)
			(bayer_grbg,V4L2_PIX_FMT_SGRBG8)
			(bayer_gbrg,V4L2_PIX_FMT_SGBRG8)

			(mjpg, 		V4L2_PIX_FMT_MJPEG)
			(jpeg, 		V4L2_PIX_FMT_JPEG)

			;


	std::map<std::string, uint32_t> special_formats=yuri::map_list_of<std::string, uint32_t>
		("S920", V4L2_PIX_FMT_SN9C20X_I420)
		("BA81", V4L2_PIX_FMT_SBGGR8);

	/** Converts yuri::format_t to v4l2 format
	 * \param fmt V4l2 pixel format
	 * \return yuri::format_t for the specified format.
	 */
	static uint32_t yuri_format_to_v4l2(yuri::format_t fmt)
	{
		if (formats_map.count(fmt)) return formats_map[fmt];
		return 0;
//		throw exception::Exception("Unknown format");
	}
	/** Converts v4l2 format to yuri::format_t
	 * \param fmt Pixel format as yuri::format_t
	 * \return v4l2 pixel format for the specified format.
	 */
	static yuri::format_t v4l2_format_to_yuri(uint32_t fmt)
	{
		for (const auto& f: formats_map) {
			if (f.second==fmt) return f.first;
		}
		return core::raw_format::unknown;
		//	case V4L2_PIX_FMT_SN9C20X_I420:	return YURI_FMT_YUV420_PLANAR;
//		throw exception::Exception("Unknown format");
	}
}

core::Parameters V4l2Source::configure()
{
	core::Parameters p = IOThread::configure();
//	p["width"]["Width of the input image. Note that actual resolution from camera may differ."]=640;
//	p["height"]["Height of the input image. Note that actual resolution from camera may differ."]=480;
	p["resolution"]["Resolution of the image. Note that actual resolution may differ"]=resolution_t{640,480};
	p["path"]["Path to the camera device. usually /dev/video0 or similar."]=std::string();
	p["method"]["Method used to get images from camera. Possible values are: none, mmap, user, read. For experts only"]="none";
	std::string fmts;
//	for (const auto& f: formats_map) {
////	std::pair<yuri::format_t,yuri::uint_t> f;
////	BOOST_FOREACH(f,formats_map) {
//		FormatInfo_t pf = core::BasicPipe::get_format_info(f.first);
//		if (!pf) continue;
//		if (!fmts.empty()) fmts+=std::string(", ");
//		fmts+=pf->short_names[0];
//	}

	p["format"][std::string("Format to capture in. Possible values are (")+fmts+")"]="YUV422";
	p["input"]["Input number to tune"]=0;
	p["illumination"]["Enable illumination (if present)"]=true;
	p["combine"]["Combine frames (if camera sends them in chunks)."]=false;
	p["fps"]["Number of frames per secod requested. The closes LOWER supported value will be selected."]=30;
	return p;
}


V4l2Source::V4l2Source(log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters)
	:core::IOThread(log_,parent,0,1,std::string("v4l2")),resolution({640,480}),
	 method(METHOD_NONE),buffers(0),no_buffers(0),buffer_free(0),
	 combine_frames(false),number_of_inputs(0),frame_duration(0)
{
	IOTHREAD_INIT(parameters)
	try {
		open_file();
		// Query capabilities for video capture
		query_capabilities();
		// Enum inputs and print them to the log
		enum_inputs();
		// Set requested input and exit if it can't be set
		set_input();
		// Set cropping (currently only full image)
		set_cropping();
		// Enumerate input formats and print them to the log
		enum_formats();
		// Set input format
		set_format();

		enum_frame_intervals();

		set_frame_params();
		// Initialize capture
		initialize_capture();
		// Enable illuminator, if requested by user
		enable_iluminator();

	}
	catch (exception::Exception &e)
	{
		throw exception::InitializationFailed(e.what());
	}
}

V4l2Source::~V4l2Source() noexcept{
	if (fd>0) close(fd);
	if (buffers) {
		if (method==METHOD_READ) for (unsigned int i=0;i<no_buffers;++i) delete [] (uint8_t*)buffers[0].start;
		else if (method==METHOD_MMAP) for (unsigned int i=0;i<no_buffers;++i) munmap(buffers[0].start,buffers[0].length);
		else if (method==METHOD_USER) for (unsigned int i=0;i<no_buffers;++i) free(buffers[0].start);
		delete [] buffers;
		buffers=0;
	}
}


void V4l2Source::run()
{
//	IO_THREAD_PRE_RUN
	fd_set set;
	struct timeval tv;
	int res=0;
	while (!start_capture()) {
		sleep(get_latency());
		if (!still_running()) break;
	}
//	int frames=0;
	while (/*frames++<1000 &&*/ still_running()) {
		FD_ZERO(&set);
		FD_SET(fd,&set);
		tv.tv_sec=0;
		tv.tv_usec=get_latency().value;///1000;
		res=select(fd+1,&set,0,0,&tv);
		if (res<0) {
			if (errno == EAGAIN || errno == EINTR) continue;
			log[log::error] << "Read error in select (" << strerror(errno)
							<< ")";
			break;
		}
		if (!res) continue;
		/// @BUG: This is crucial, otherwise there's some race cond... ;/
		log[log::verbose_debug] << "Reading!";
		if (!read_frame()) break;

		log[log::verbose_debug] << "Frame!";
	}
	log[log::info] << "Stopping capture";
	stop_capture();
//	IO_THREAD_POST_RUN
}

bool V4l2Source::step()
{
	return false;
}
bool V4l2Source::init_mmap()
{
	struct v4l2_requestbuffers req;
	memset(&req,0,sizeof(req));
	req.count = 4;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_MMAP;

	if (xioctl (fd, VIDIOC_REQBUFS, &req)<0) {
		if (errno == EINVAL) {
			log[log::warning] << "Device does not support memory mapping";
	            return false;
		} else {
			log[log::warning] << "VIDIOC_REQBUFS failed";
			return false;
		}
	}
	if (req.count < 2) {
		log[log::warning] << "Insufficient buffer memory";
		return false;
	}
	no_buffers=req.count;
	buffers=new buffer_t[no_buffers];
	for (unsigned int i=0; i < req.count; ++i) {
		struct v4l2_buffer buf;
		memset(&buf,0,sizeof(buf));
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	    buf.memory = V4L2_MEMORY_MMAP;
	    buf.index = i;
	    if (xioctl (fd, VIDIOC_QUERYBUF, &buf)<0) {
	    	log[log::error] << "VIDIOC_QUERYBUF failed. (" << strerror(errno)
							<< ")";
	    	return false;
	    }
		buffers[i].length = buf.length;
		buffers[i].start = mmap (NULL, buf.length,
				PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

		if (buffers[i].start == MAP_FAILED) {
				log[log::error] << "mmap failed (" << errno << ") - "
					<< strerror(errno);
				return false;
		}
	}
	return true;
}

bool V4l2Source::init_user()
{
	struct v4l2_requestbuffers req;
	unsigned int page_size;
	unsigned long buffer_size=imagesize;
	page_size = getpagesize ();
	buffer_size = (buffer_size + page_size - 1) & ~(page_size - 1);

	memset(&req,0,sizeof(req));

	req.count = 4;
	req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	req.memory = V4L2_MEMORY_USERPTR;
	if (xioctl (fd, VIDIOC_REQBUFS, &req)<0) {
		if (errno == EINVAL) {
			log[log::warning] << "Device does not support user pointers"
				<< std::endl;
	            return false;
		} else {
			log[log::warning] << "VIDIOC_REQBUFS failed (" << strerror(errno)
							<< ")" << std::endl;
			return false;
		}
	}
	if (req.count < 2) {
		log[log::warning] << "Insufficient buffer memory" << std::endl;
		return false;
	}
	no_buffers=req.count;
	buffers=new buffer_t[no_buffers];
	for (unsigned int i=0; i < req.count; ++i) {
		buffers[i].length = buffer_size;
		buffers[i].start = memalign (page_size, buffer_size);

		if (!buffers[i].start) {
				log[log::error] << "Out of memory failed" << std::endl;
				return false;
		}
	}
	return true;
}

bool V4l2Source::init_read()
{
	no_buffers=1;
	buffers=new buffer_t[1];
	buffers[0].length=imagesize;
	buffers[0].start=new uint8_t[imagesize];
	return true;
}

bool V4l2Source::start_capture()
{

	enum v4l2_buf_type type;
	switch (method) {
		case METHOD_READ: return true;
		case METHOD_MMAP:
			for (unsigned int i=0; i < no_buffers; ++i) {
				if (buffers[i].start == MAP_FAILED) {
						log[log::error] << "mmap failed (" << errno << ") - "
							<< strerror(errno) << std::endl;
						return false;
				}
				struct v4l2_buffer buf;
				memset (&buf,0,sizeof(buf));
				buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				buf.memory      = V4L2_MEMORY_MMAP;
				buf.index       = i;
				if (xioctl (fd, VIDIOC_QBUF, &buf) == -1) {
					log[log::error] << "VIDIOC_QBUF failed (" << strerror(errno)
							<< ")" << std::endl;
					return false;
				}
			}
			type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			if (xioctl(fd, VIDIOC_STREAMON, &type)==-1) {
				log[log::error] << "VIDIOC_STREAMON failed (" << strerror(errno)
										<< ")" << std::endl;
								return false;
			}
			return true;
		case METHOD_USER:
			for (unsigned int i = 0; i < no_buffers; ++i) {
				struct v4l2_buffer buf;
				memset (&buf,0,sizeof(buf));
				buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				buf.memory      = V4L2_MEMORY_USERPTR;
				buf.index       = i;
				buf.m.userptr   = (unsigned long) buffers[i].start;
				buf.length      = buffers[i].length;
				if (xioctl (fd, VIDIOC_QBUF, &buf) == -1) {
					log[log::error] << "VIDIOC_QBUF failed (" << strerror(errno)
							<< ")" << std::endl;
					return false;
				}
				type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				if (xioctl (fd, VIDIOC_STREAMON, &type)==-1) {
					log[log::error] << "VIDIOC_STREAMON failed (" << strerror(errno)
							<< ")" << std::endl;
					return false;
				}
			}
			return true;
		case METHOD_NONE:

		default: return false;
	}
}

bool V4l2Source::read_frame()
{
	int res=0;
	struct v4l2_buffer buf;
//	core::pRawVideoFrame frame;
	switch (method) {
		case METHOD_READ:
					res=read(fd,buffers[0].start,imagesize);
					if (res<0) {
						if (errno==EAGAIN || errno==EINTR) return true;
						log[log::error] << "Read error (" << errno << ") - " << strerror(errno);
						return false;
					}
					if (!res) return true; // Should never happen
					prepare_frame(reinterpret_cast<uint8_t*>(buffers[0].start),
							imagesize);
					break;
		case METHOD_MMAP:
					memset(&buf,0,sizeof(buf));
					buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
					buf.memory = V4L2_MEMORY_MMAP;
					if (xioctl (fd, VIDIOC_DQBUF, &buf)==-1) {
						switch (errno) {
							case EAGAIN:
									return true;
							case EIO:
							default:
									log[log::error] << "VIDIOC_DQBUF failed (" <<
										strerror(errno) << ")";
									//start_capture();
									for (size_t i = 0; i < no_buffers; ++i) {
										memset(&buf,0,sizeof(buf));
										buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
										buf.memory = V4L2_MEMORY_MMAP;
										buf.index = i;
										if ( xioctl(fd,VIDIOC_QUERYBUF,&buf) < 0) {
											log[log::error] << "Failed to query buffer " << i << "(" << strerror(errno);
											continue;
										}
										if ((buf.flags & (V4L2_BUF_FLAG_QUEUED | V4L2_BUF_FLAG_MAPPED | V4L2_BUF_FLAG_DONE)) == V4L2_BUF_FLAG_MAPPED) {
											if (xioctl(fd,VIDIOC_QBUF,&buf)<0) {
												log[log::error] << "Failed to queue buffer " << i << "(" << strerror(errno);
												return false;
											}
										}
									}

									return true;
						}
					}
					if (buf.index >= no_buffers) {
						log[log::error] << "buf.index >= n_buffers!!!!";
						return false;
					}
					//if (!out[0].get()) return true;
					log[log::verbose_debug] << "Pushing frame with " << buf.bytesused
							<< "bytes";
					prepare_frame(reinterpret_cast<uint8_t*>(buffers[buf.index].start),buf.bytesused);
					if (xioctl (fd, VIDIOC_QBUF, &buf)==-1) {
							log[log::error] << "VIDIOC_QBUF failed";
							return false;
					}
					break;
					//return true;
		case METHOD_USER:
					memset(&buf,0,sizeof(buf));
					buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
					buf.memory = V4L2_MEMORY_MMAP;
					if (xioctl (fd, VIDIOC_DQBUF, &buf)==-1) {
						switch (errno) {
							case EAGAIN:
									return true;
							case EIO:
							default:
									log[log::error] << "VIDIOC_DQBUF failed";
									return false;
							}
					}
					prepare_frame(reinterpret_cast<uint8_t*>(buf.m.userptr), buf.length);
					if (xioctl (fd, VIDIOC_QBUF, &buf)==-1) {
							log[log::error] << "VIDIOC_QBUF failed";
							return false;
					}
					break;
		case METHOD_NONE:
		default: return false;
	}
	if (output_frame && !buffer_free) {
//		output_frame->set_time(0,0,frame_duration);
//		if (out[0]) push_raw_video_frame(0,timestamp_frame(output_frame));
		push_frame(0, output_frame);
		output_frame.reset();
	}
	return true;
}

bool V4l2Source::stop_capture()
{
	enum v4l2_buf_type type;
	switch (method) {
		case METHOD_READ: return true;
		case METHOD_MMAP:
		case METHOD_USER:
			type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			if (xioctl (fd, VIDIOC_STREAMOFF, &type)==-1) {
				log[log::error] << "VIDIOC_STREAMOFF failed";
				return false;
			}
			return true;
		case METHOD_NONE:
		default: return false;
	}
}

int V4l2Source::xioctl(int fd, unsigned long int request, void *arg)
{
	int r;
	while((r = ioctl (fd, request, arg)) < 0) {
		if (r==-1 && errno==EINTR) continue;
		break;
	}
	return r;
}


bool V4l2Source::set_param(const core::Parameter &param)
{
	log[log::info] << "Processing param " << param.get_name() << " = " << param.get<std::string>();
	if (param.get_name() == "path") {
		filename = param.get<std::string>();
	} else if (param.get_name() == "format") {
		std::string format = param.get<std::string>();
		format_t fmt = core::raw_format::parse_format(format);
		if (!fmt) {
			log[log::info] << "Specified not-raw format";
			fmt = core::compressed_frame::parse_format(format);
			log[log::info] << "Format parsed as: " << fmt;
		}

		pixelformat = yuri_format_to_v4l2(fmt);
		if (!pixelformat) {
			// Process special formats....
//			else if (boost::iequals(format,"S920")) pixelformat = V4L2_PIX_FMT_SN9C20X_I420;
//			else if (boost::iequals(format,"BA81")) pixelformat = V4L2_PIX_FMT_SBGGR8;
			log[log::warning] << "Unsupported format specified. Trying YUV422";
			pixelformat = V4L2_PIX_FMT_YUYV;
		}

//	} else if (param.get_name() == "width") {
//		width = param.get<yuri::size_t>();
//	} else if (param.get_name() == "height") {
//		height = param.get<yuri::size_t>();
	} else if (param.get_name() == "resolution") {
		resolution = param.get<resolution_t>();
	} else if (param.get_name() == "method") {
		std::string method_s;
		method_s = param.get<std::string>();
		if (iequals(method_s,"user")) method = METHOD_USER;
		else if (iequals(method_s,"mmap")) method = METHOD_MMAP;
		else if (iequals(method_s,"read")) method = METHOD_READ;
		else method=METHOD_NONE;
	} else if (param.get_name() == "input") {
		input_number = param.get<size_t>();
	} else if (param.get_name() == "illumination") {
			illumination = param.get<bool>();
	} else if (param.get_name() == "combining") {
		combine_frames = param.get<bool>();
	} else if (param.get_name() == "fps") {
		fps= param.get<yuri::size_t>();
	} else return IOThread::set_param(param);
	return true;

}
bool V4l2Source::prepare_frame(uint8_t *data, yuri::size_t size)
{
	yuri::format_t fmt = v4l2_format_to_yuri(pixelformat);
	if (!fmt) return false;

	try {
		const raw_format_t& fi = core::raw_format::get_format_info(fmt);
		size_t frame_size = resolution.width*resolution.height*fi.planes[0].bit_depth.first/fi.planes[0].bit_depth.second/8;

		core::pRawVideoFrame rframe = dynamic_pointer_cast<core::RawVideoFrame>(output_frame);
		if (!rframe) {
			rframe = core::RawVideoFrame::create_empty(fmt, resolution, true);
			buffer_free = frame_size;
			output_frame = rframe;
		}
		yuri::size_t frame_position = frame_size - buffer_free;
		log[log::verbose_debug] << "Frame " << resolution.width << ", " << resolution.height << ", size: " << size;
		if (fi.planes.size()==1) {
			if (size>buffer_free) size = buffer_free;
			std::copy(data, data + size, PLANE_DATA(rframe, 0).begin());
			buffer_free -= size;
		} else {
			yuri::size_t offset = 0;
			for (yuri::size_t i = 0; i < fi.planes.size(); ++i) {
				if (!size) break;
				yuri::size_t cols = resolution.width / fi.planes[i].sub_x;
				yuri::size_t rows = resolution.height / fi.planes[i].sub_y;
				yuri::size_t plane_size = (cols*rows*fi.planes[i].bit_depth.first/fi.planes[i].bit_depth.second)>>3;
				//if(size<offset+plane_size) return pBasicFrame();
				if (plane_size > frame_position) {
					plane_size -= frame_position;
					frame_position = 0;
				} else {
					frame_position-=plane_size;
					continue;
				}
				if(size<offset+plane_size) {
					plane_size = size-offset;
				}
				if (plane_size > buffer_free) {
					plane_size = buffer_free;
				}
				log[log::info] << "Copying " << plane_size << " bytes, have " << size-offset <<", free buffer: " << buffer_free<< std::endl;
				std::copy(data+offset, data+offset+plane_size, PLANE_DATA(rframe, i).begin());
				//memcpy(PLANE_RAW_DATA(output_frame,i),data+offset,plane_size);
				offset+=plane_size;
				buffer_free-=plane_size;
			}
		}
	}
	catch (std::runtime_error& ) {
		core::pCompressedVideoFrame cframe = core::CompressedVideoFrame::create_empty(fmt, resolution, data, size);
		buffer_free = 0;//frame_size;
		output_frame = cframe;


	}

	// If we're no combining frames, we have to discard incomplete ones
	if (buffer_free && !combine_frames) {

		log[log::warning] << "Discarding incomplete frame (missing " << buffer_free << " bytes)";
		buffer_free = 0;
		output_frame.reset();
	}
	return true;
}

bool V4l2Source::open_file()
{

	if (filename.empty()) throw exception::Exception("Path must be specified!");
	//try {
	fd=open(filename.c_str(),O_RDWR|O_NONBLOCK);
	if (fd<0) {
		log[log::error] << "Failed to open file " << filename << std::endl;
		throw exception::Exception("Failed to open file "+filename);
	}
	log[log::info] << filename << " opened successfully";
	//}
//	catch (boost::sy)

	return true;

}
bool V4l2Source::query_capabilities()
{
	if (xioctl(fd,VIDIOC_QUERYCAP,&cap)<0) {
		log[log::error] << "VIDIOC_QUERYCAP ioctl failed!" << std::endl;
		throw exception::Exception("VIDIOC_QUERYCAP ioctl failed!");
	}
	log[log::info]<< "Using driver: " << cap.driver << ", version " << ((cap.version >> 16) & 0xFF) << "." << ((cap.version >> 8) & 0xFF )<< "." <<  (cap.version & 0xFF) << std::endl;
	log[log::info]<< "Card name: " << cap.card << ", connected to: " << cap.bus_info << std::endl;
	if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE ) log[log::info] << "Device supports Video capture" << std::endl;
	else {
		log[log::error] << "Device does not supports Video capture!" << std::endl;
		throw exception::Exception("Device does not support video capture!");
	}
	return true;
}
bool V4l2Source::enum_inputs()
{
	v4l2_input input_info;
	input_info.index=0;
	while (xioctl(fd,VIDIOC_ENUMINPUT,&input_info)) {
		log[log::info] << "Input " << input_info.index << ": " << input_info.name <<
				", type: " << (input_info.type==V4L2_INPUT_TYPE_CAMERA?"camera":"tuner")
				<< ", status: " << (!input_info.status?"OK":(input_info.status==V4L2_IN_ST_NO_POWER?"No power":(input_info.status==V4L2_IN_ST_NO_SIGNAL?"No signal":"No color")))
				<< std::endl;
		input_info.index++;
	}
	number_of_inputs = input_info.index;
	return true;
}
bool V4l2Source::set_input()
{
	// Not checking support for input 0 - many webcams does not report any input even thoug they have input 0
	if (input_number && input_number >= number_of_inputs) {
		// User is trying to set input not supported by the device. Let's only warn him here, the error will be returned later
		log[log::warning] << "The device reports that it does not support requested input "
				<< input_number << ". Trying to set it anyway" << std::endl;
	}
	log[log::debug] << "Setting input to " << input_number << std::endl;
	int inp = input_number;
	if (!xioctl (fd, VIDIOC_S_INPUT, &inp)) {
		log[log::debug] <<"VIDIOC_S_INPUT failed, input was NOT set. " << std::endl;
		// Let's assume that default input is 0. So not being able to set 0 is not really an error
		if (input_number) throw exception::Exception("Failed to set input ");
	} else {
		log[log::info] << "Input set to " << input_number << std::endl;
	}
	return true;

}
bool V4l2Source::set_cropping()
{
	v4l2_cropcap cropcap;
	v4l2_crop crop;
	memset(&cropcap,0,sizeof(cropcap));
	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (!xioctl (fd, VIDIOC_CROPCAP, &cropcap)) {
		crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		crop.c = cropcap.defrect; /* reset to default */
		log[log::info] << "Selected input have pixel with aspect ratio " <<
				cropcap.pixelaspect.numerator << "/" << cropcap.pixelaspect.denominator << std::endl;
		if (xioctl (fd, VIDIOC_S_CROP, &crop) == -1) {
			log[log::warning] <<"VIDIOC_S_CROP failed, ignoring :)" << std::endl;
		}
	} else {
		log[log::warning] << "Failed to query cropping info, ignoring" << std::endl;
	}
	return true;
}
bool V4l2Source::enum_formats()
{
	v4l2_fmtdesc fmts;
	supported_formats.clear();
	fmts.index=0;
	fmts.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
	while (!xioctl(fd,VIDIOC_ENUM_FMT,&fmts)) {
		auto l = log[log::info];
		l << "Supported format " << fmts.index << ": " << fmts.description;
		try {
			format_t fmt = v4l2_format_to_yuri(fmts.pixelformat);
			const auto& fi = core::raw_format::get_format_info(fmt);
			if (fi.short_names.size() > 0)
			l << " [yuri fmt: " << fi.short_names[0] << "]";
		} catch (std::exception&) {}
		fmts.index++;
		supported_formats.push_back(fmts.pixelformat);
	}
	return true;
}
bool V4l2Source::set_format()
{
	memset (&fmt,0,sizeof(v4l2_format));
	fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (xioctl(fd,VIDIOC_G_FMT,&fmt)<0) {
		log[log::error] << "VIDIOC_G_FMT ioctl failed! (" << strerror(errno)
							<< ")" << std::endl;
		throw exception::Exception("Failed to get default format info!");
	}
	fmt.fmt.pix.pixelformat=pixelformat;
	fmt.fmt.pix.width=resolution.width;
	fmt.fmt.pix.height=resolution.height;
	if (xioctl(fd,VIDIOC_S_FMT,&fmt)<0) {
		log[log::error] << "VIDIOC_S_FMT ioctl failed!";
		throw exception::Exception ("Failed to set input format!");
	}
	if (xioctl(fd,VIDIOC_G_FMT,&fmt)<0) {
		log[log::warning] << "Failed to verify if input format was set correctly !";
	}
	if (fmt.fmt.pix.pixelformat != pixelformat) {
		log[log::error] << "Failed to set requested input format!";
		throw exception::Exception("Failed to set input format");
	}
	log[log::info] << "Video dimensions: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;
	log[log::info] << "Pixel format (" << fmt.fmt.pix.pixelformat << "): " <<
		(char)(fmt.fmt.pix.pixelformat & 0xFF) <<
		(char)(fmt.fmt.pix.pixelformat>>8 & 0xFF) <<
		(char)(fmt.fmt.pix.pixelformat>>16 & 0xFF) <<
		(char)(fmt.fmt.pix.pixelformat>>24 & 0xFF)
		<< std::endl;
	log[log::info] << "Colorspace " << fmt.fmt.pix.colorspace;
	imagesize=fmt.fmt.pix.sizeimage;
	resolution.width=fmt.fmt.pix.width;
	resolution.height=fmt.fmt.pix.height;
	pixelformat=fmt.fmt.pix.pixelformat;

	log[log::info] << "Image size: " << imagesize << std::endl;
	return true;
}
bool V4l2Source::enum_frame_intervals()
{
	v4l2_frmivalenum frmvalen;
	frmvalen.pixel_format = pixelformat;
	frmvalen.width = resolution.width;
	frmvalen.height = resolution.height;
	frmvalen.index = 0;
	while (!xioctl(fd,VIDIOC_ENUM_FRAMEINTERVALS,&frmvalen)) {
		switch (frmvalen.type) {
		case V4L2_FRMIVAL_TYPE_CONTINUOUS:
			log[log::info] << "Supports continuous frame_intervals from"
				<< frmvalen.stepwise.min.numerator << "/" << frmvalen.stepwise.min.denominator
				<< "s to " << frmvalen.stepwise.max.numerator << "/" << frmvalen.stepwise.max.denominator <<"s"<< std::endl;
			break;
		case V4L2_FRMIVAL_TYPE_STEPWISE:
			log[log::info] << "Supports stepwise frame_intervals from"
				<< frmvalen.stepwise.min.numerator << "/" << frmvalen.stepwise.min.denominator
				<< "s to " << frmvalen.stepwise.max.numerator << "/" << frmvalen.stepwise.max.denominator
				<< "s with step " << frmvalen.stepwise.step.numerator << "/" << frmvalen.stepwise.step.denominator<<"s"<< std::endl;
			break;
		case V4L2_FRMIVAL_TYPE_DISCRETE:
			if (!frmvalen.index) log[log::info] << "Supports discrete frame_intervals:" << std::endl;
			log[log::info] << "\t"<<frmvalen.index<<": "<< frmvalen.discrete.numerator << "/" << frmvalen.discrete.denominator << "s" << std::endl;
			frmvalen.index++;
			break;
		}
		if (!frmvalen.index) break;
		//log[log::info] << "Supported frame_interval " << fmts.index << ": " << fmts.description << std::endl;

		//supported_formats.push_back(fmts.pixelformat);
	}
	return true;
}
bool V4l2Source::set_frame_params()
{
	v4l2_streamparm strp;
	strp.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
	strp.parm.capture.capability=V4L2_CAP_TIMEPERFRAME;
	strp.parm.capture.readbuffers=0;
	strp.parm.capture.extendedmode=0;
	strp.parm.capture.capturemode=0;
	strp.parm.capture.timeperframe.numerator=1;
	strp.parm.capture.timeperframe.denominator=fps;
	if (xioctl(fd,VIDIOC_S_PARM,&strp)<0) {
		log[log::error] << "Failed to set frame parameters (FPS)" << std::endl;
		//throw exception::Exception ("Failed to set input format!");
		return false;
	}
	if (xioctl(fd,VIDIOC_G_PARM,&strp)<0) {
		log[log::error] << "Failed to verify frame parameters (FPS)" << std::endl;
		//throw exception::Exception ("Failed to set input format!");
		return false;
	}
	log[log::info] << "Driver reports current frame interval " << strp.parm.capture.timeperframe.numerator << "/"
			<< strp.parm.capture.timeperframe.denominator << "s" << std::endl;
	frame_duration = 1e6*strp.parm.capture.timeperframe.numerator/strp.parm.capture.timeperframe.denominator;
	return true;
}
bool V4l2Source::initialize_capture()
{
	if (cap.capabilities & V4L2_CAP_STREAMING) {
		log[log::debug] << "Driver supports streaming operations, trying to initialize" << std::endl;
		if ((method==METHOD_NONE || method == METHOD_MMAP) && init_mmap()) {
			log[log::info] << "Initialized mmap " << std::endl;
			method=METHOD_MMAP;
		} else {
			log[log::debug] << "mmap failed, trying user pointers" << std::endl;
			if ((method==METHOD_NONE || method == METHOD_USER) && init_user()) {
				log[log::info] << "Initialized capture using user pointers" << std::endl;
				method=METHOD_USER;
			} else {
				log[log::debug] << "user pointers failed." << std::endl;
				method=METHOD_NONE;
			}
		}
	}
	else {
		log[log::debug] << "Driver does not support streaming operations!" << std::endl;
	}
	if(cap.capabilities & V4L2_CAP_READWRITE) {
		log[log::debug] << "Driver supports read/write operations" << std::endl;
		if (method==METHOD_NONE) {
			if (init_read()) {
				log[log::info] << "Initialized direct reading from device file" <<std::endl;
				method=METHOD_READ;
			}
		}
	}
	else {
		log[log::debug] << "Driver does not support read/write operations!" << std::endl;
	}
	if (method==METHOD_NONE) {
		log[log::fatal] << "I do not know how to read from this camera!" << std::endl;
		throw exception::Exception("I do not know how to read from this camera!!");
	}
	return true;
}


/**
 * \brief Enables illuminator
 *
 * If illuminator is requested, method tries to enable it
 * \return true if successfull
 * \throw yuri::exception::Exception if there's any fatal error (currently there's none)
 */
bool V4l2Source::enable_iluminator()
{
	if (illumination) {
		struct v4l2_queryctrl queryctrl;
		struct v4l2_control control;
		memset (&queryctrl, 0, sizeof (queryctrl));
		queryctrl.id=V4L2_CID_ILLUMINATORS_1;

		if (xioctl (fd, VIDIOC_QUERYCTRL, &queryctrl)<0) {
		        log[log::error] << "Illuminator is not supported" << std::endl;
		} else if (queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) {
			log[log::error] << "Illuminator is disabled" << std::endl;
		} else {
			log[log::debug] << "Trying to enable illuminator " << queryctrl.name << std::endl;
			memset (&control, 0, sizeof (control));
			control.id = V4L2_CID_ILLUMINATORS_1;
			control.value = queryctrl.maximum;
			if (xioctl (fd, VIDIOC_S_CTRL, &control)<0) {
				log[log::error]<< "Failed to enable illuminator" << std::endl;
			} else {
				control.value = 0;
				if (xioctl (fd, VIDIOC_G_CTRL, &control)>=0) {
					if (control.value) {
						log[log::info] << "Illuminator enabled." <<std::endl;
					} else {
						log[log::error] << "Illuminator set, but camera reports it's not..." << std::endl;
					}
				} else {
					log[log::info]<<"Illuminator enabled, but failed to query it's status."<<std::endl;
				}
			}
		}
	}
	return true;
}

}
}

