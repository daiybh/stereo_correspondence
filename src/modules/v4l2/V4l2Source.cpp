/*!
 * @file 		V4l2Source.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		17.5.2009
 * @date		25.1.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2009 - 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
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

#include <errno.h>
#include <sys/ioctl.h>

namespace yuri {

namespace v4l2 {



IOTHREAD_GENERATOR(V4l2Source)
MODULE_REGISTRATION_BEGIN("v4l2source")
	REGISTER_IOTHREAD("v4l2source",V4l2Source)
MODULE_REGISTRATION_END()

namespace {
	int xioctl(int fd, unsigned long int request, void *arg)
	{
		int r;
		while((r = ioctl (fd, request, arg)) < 0) {
			if (r==-1 && errno==EINTR) continue;
			break;
		}
		return r;
	}

}


#include "v4l2_constants.cpp"

namespace {
V4l2Source::methods parse_method(const std::string& method_s)
{
	if (iequals(method_s,"user")) return V4l2Source::METHOD_USER;
	else if (iequals(method_s,"mmap")) return V4l2Source::METHOD_MMAP;
	else if (iequals(method_s,"read")) return V4l2Source::METHOD_READ;
	else return V4l2Source::METHOD_NONE;
}
}

core::Parameters V4l2Source::configure()
{
	core::Parameters p = IOThread::configure();
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
	p["fps"]["Number of frames per secod requested. The closes LOWER supported value will be selected."]=fraction_t{30,1};
	return p;
}


V4l2Source::V4l2Source(log::Log &log_,core::pwThreadBase parent, const core::Parameters &parameters)
	:core::IOThread(log_,parent,0,1,std::string("v4l2")),
	 event::BasicEventConsumer(log),
	 resolution({640,480}),
	 method(METHOD_NONE),buffers(0),no_buffers(0),buffer_free(0),
	 combine_frames(false),number_of_inputs(0),fps{30,1},frame_duration(0)
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

		enum_controls();


	}
	catch (exception::Exception &e)
	{
		throw exception::InitializationFailed(e.what());
	}
}

V4l2Source::~V4l2Source() noexcept{
	if (fd>0) close(fd);
	if (buffers) {
		if (method==METHOD_READ) for (unsigned int i=0;i<no_buffers;++i) delete [] static_cast<uint8_t*>(buffers[i].start);
		else if (method==METHOD_MMAP) for (unsigned int i=0;i<no_buffers;++i) munmap(buffers[i].start,buffers[i].length);
		else if (method==METHOD_USER) for (unsigned int i=0;i<no_buffers;++i) delete[] static_cast<uint8_t*>(buffers[i].start);
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
	while (still_running()) {
		process_events();
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
		output_frame->set_duration(frame_duration);
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




bool V4l2Source::set_param(const core::Parameter &param)
{
	if (assign_parameters(param)
			(filename, "path")
			(resolution, "resolution")
			(input_number, "input")
			(illumination, "illumination")
			(combine_frames, "combine")
			(fps, "fps")
			(method, "method", [](const core::Parameter& p){return parse_method(p.get<std::string>());})
			)
		return true;

	if (param.get_name() == "format") {
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
	// Numerator and denominator are switched because we're changing it from FPS to frame time...
	strp.parm.capture.timeperframe.numerator=fps.denom;
	strp.parm.capture.timeperframe.denominator=fps.num;
	if (xioctl(fd,VIDIOC_S_PARM,&strp)<0) {
		log[log::error] << "Failed to set frame parameters (FPS)";
		//throw exception::Exception ("Failed to set input format!");
		return false;
	}
	if (xioctl(fd,VIDIOC_G_PARM,&strp)<0) {
		log[log::error] << "Failed to verify frame parameters (FPS)";
		//throw exception::Exception ("Failed to set input format!");
		return false;
	}
	fps = {strp.parm.capture.timeperframe.denominator,strp.parm.capture.timeperframe.numerator};
	log[log::info] << "Driver reports current frame interval " << !fps << "s";
	if (fps.valid() || fps.num!=0) {
		frame_duration = 1_s/fps.get_value();
	} else {
		frame_duration = 0_s;
	}
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
	return controls::set_control(fd, "iluminator", illumination, log);
}

bool V4l2Source::enum_controls()
{
	controls_ = controls::get_control_list(fd, log);
	log[log::info] << "Supported controls:";
	for (const auto& c: controls_) {
		log[log::info] << "\t'" << c.name << "' (" << c.short_name
				<< "), value: " << c.value
				<< ", range: <" << c.min_value<<", "<<c.max_value<<">";
	}
	return true;
}

bool V4l2Source::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	auto it = std::find_if(controls_.cbegin(), controls_.cend(), [&event_name](const controls::control_info&info){return iequals(info.name, event_name);});
	if (it != controls_.cend()) {
		controls::set_control(fd, it->id, event, log);
	} else if (!controls::set_control(fd, event_name, event, log)) {
//		log[log::warning] << "set control failed " << event_name;
	}
	return true;
}

}
}

