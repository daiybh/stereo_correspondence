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
#include <boost/algorithm/string.hpp>
#include <yuri/config/RegisteredClass.h>
#include <string>
#include <boost/assign.hpp>
namespace yuri {

namespace io {


REGISTER("v4l2source",V4l2Source)

shared_ptr<BasicIOThread> V4l2Source::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<V4l2Source> v4l2 (new V4l2Source(_log,parent,parameters));
	return v4l2;
}
shared_ptr<Parameters> V4l2Source::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	(*p)["width"]["Width of the input image. Note that actual resolution from camera may differ."]=640;
	(*p)["height"]["Height of the input image. Note that actual resolution from camera may differ."]=480;
	(*p)["path"]["Path to the camera device. usially /dev/video0 or similar."]=std::string();
	(*p)["method"]["Method used to get images from camera. Possible values are: none, mmap, user, read. For experts only"]="none";
std::string fmts;
	std::pair<yuri::format_t,yuri::uint_t> f;
	BOOST_FOREACH(f,formats_map) {
		FormatInfo_t pf = BasicPipe::get_format_info(f.first);
		if (!pf) continue;
		if (!fmts.empty()) fmts+=std::string(", ");
		fmts+=pf->short_names[0];
	}

	(*p)["format"][std::string("Format to capture in. Possible values are (")+fmts+")"]="YUV422";
	(*p)["input"]["Input number to tune"]=0;
	(*p)["illumination"]["Enable illumination (if present)"]=true;
	(*p)["combine"]["Combine frames (if camera sends them in chunks)."]=false;
	(*p)["fps"]["Number of frames per secod requested. The closes LOWER supported value will be selected."]=0;
	return p;
}

std::map<yuri::format_t, yuri::uint_t> V4l2Source::formats_map=
		boost::assign::map_list_of<yuri::format_t,yuri::uint_t>
		(YURI_FMT_RGB,V4L2_PIX_FMT_RGB24)
		(YURI_FMT_RGBA, V4L2_PIX_FMT_RGB32)
		(YURI_FMT_BGR, V4L2_PIX_FMT_BGR24)
		(YURI_FMT_BGRA, V4L2_PIX_FMT_BGR32)
		(YURI_FMT_YUV422, V4L2_PIX_FMT_YUYV)
		(YURI_FMT_YUV420_PLANAR, V4L2_PIX_FMT_YUV420)
		(YURI_VIDEO_DV, V4L2_PIX_FMT_DV)
		(YURI_VIDEO_MJPEG, V4L2_PIX_FMT_MJPEG)
		(YURI_IMAGE_JPEG, V4L2_PIX_FMT_JPEG);


std::map<std::string, yuri::uint_t> V4l2Source::special_formats=boost::assign::map_list_of<std::string,yuri::uint_t>
	("S920", V4L2_PIX_FMT_SN9C20X_I420)
	("BA81", V4L2_PIX_FMT_SBGGR8);

V4l2Source::V4l2Source(Log &log_,pThreadBase parent,Parameters &parameters)
	IO_THREAD_CONSTRUCTOR
	:BasicIOThread(log_,parent,0,1,std::string("v4l2")),
	 method(METHOD_NONE),buffers(0),no_buffers(0),buffer_free(0),
	 combine_frames(false),number_of_inputs(0),frame_duration(0)
{
	IO_THREAD_INIT("V4l2Source")
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
	catch (Exception &e)
	{
		throw InitializationFailed(e.what());
	}
}

V4l2Source::~V4l2Source() {
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
	IO_THREAD_PRE_RUN
	fd_set set;
	struct timeval tv;
	int res=0;
	while (!start_capture()) {
		ThreadBase::sleep(latency);
		if (!still_running()) break;
	}
//	int frames=0;
	while (/*frames++<1000 &&*/ still_running()) {
		FD_ZERO(&set);
		FD_SET(fd,&set);
		tv.tv_sec=0;
		tv.tv_usec=latency;
		res=select(fd+1,&set,0,0,&tv);
		if (res<0) {
			if (errno == EAGAIN || errno == EINTR) continue;
			log[error] << "Read error in select (" << strerror(errno)
							<< ")" << std::endl;
			break;
		}
		if (!res) continue;
		if (!read_frame()) break;

		log[verbose_debug] << "Frame!" << std::endl;
	}
	stop_capture();
	IO_THREAD_POST_RUN
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
			log[warning] << "Device does not support memory mapping"
				<< std::endl;
	            return false;
		} else {
			log[warning] << "VIDIOC_REQBUFS failed" << std::endl;
			return false;
		}
	}
	if (req.count < 2) {
		log[warning] << "Insufficient buffer memory" << std::endl;
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
	    	log[error] << "VIDIOC_QUERYBUF failed. (" << strerror(errno)
							<< ")" << std::endl;
	    	return false;
	    }
		buffers[i].length = buf.length;
		buffers[i].start = mmap (NULL, buf.length,
				PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);

		if (buffers[i].start == MAP_FAILED) {
				log[error] << "mmap failed (" << errno << ") - "
					<< strerror(errno) << std::endl;
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
			log[warning] << "Device does not support user pointers"
				<< std::endl;
	            return false;
		} else {
			log[warning] << "VIDIOC_REQBUFS failed (" << strerror(errno)
							<< ")" << std::endl;
			return false;
		}
	}
	if (req.count < 2) {
		log[warning] << "Insufficient buffer memory" << std::endl;
		return false;
	}
	no_buffers=req.count;
	buffers=new buffer_t[no_buffers];
	for (unsigned int i=0; i < req.count; ++i) {
		buffers[i].length = buffer_size;
		buffers[i].start = memalign (page_size, buffer_size);

		if (!buffers[i].start) {
				log[error] << "Out of memory failed" << std::endl;
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
						log[error] << "mmap failed (" << errno << ") - "
							<< strerror(errno) << std::endl;
						return false;
				}
				struct v4l2_buffer buf;
				memset (&buf,0,sizeof(buf));
				buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				buf.memory      = V4L2_MEMORY_MMAP;
				buf.index       = i;
				if (xioctl (fd, VIDIOC_QBUF, &buf) == -1) {
					log[error] << "VIDIOC_QBUF failed (" << strerror(errno)
							<< ")" << std::endl;
					return false;
				}
			}
			type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
			if (xioctl(fd, VIDIOC_STREAMON, &type)==-1) {
				log[error] << "VIDIOC_STREAMON failed (" << strerror(errno)
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
					log[error] << "VIDIOC_QBUF failed (" << strerror(errno)
							<< ")" << std::endl;
					return false;
				}
				type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				if (xioctl (fd, VIDIOC_STREAMON, &type)==-1) {
					log[error] << "VIDIOC_STREAMON failed (" << strerror(errno)
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
	pBasicFrame frame;
	switch (method) {
		case METHOD_READ:
					res=read(fd,buffers[0].start,imagesize);
					if (res<0) {
						if (errno==EAGAIN || errno==EINTR) return true;
						log[error] << "Read error (" << errno << ") - " << strerror(errno)
							<< std::endl;
						return false;
					}
					if (!res) return true; // Should never happen
					prepare_frame(reinterpret_cast<yuri::ubyte_t*>(buffers[0].start),
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
									log[error] << "VIDIOC_DQBUF failed (" <<
										strerror(error) << ")" << std::endl;
									//start_capture();
									for (yuri::ushort_t i=0;i<no_buffers;++i) {
										memset(&buf,0,sizeof(buf));
										buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
										buf.memory = V4L2_MEMORY_MMAP;
										buf.index = i;
										if ( xioctl(fd,VIDIOC_QUERYBUF,&buf) < 0) {
											log[error] << "Failed to query buffer " << i << "(" << strerror(errno) << std::endl;
											continue;
										}
										if ((buf.flags & (V4L2_BUF_FLAG_QUEUED | V4L2_BUF_FLAG_MAPPED | V4L2_BUF_FLAG_DONE)) == V4L2_BUF_FLAG_MAPPED) {
											if (xioctl(fd,VIDIOC_QBUF,&buf)<0) {
												log[error] << "Failed to queue buffer " << i << "(" << strerror(errno) << std::endl;
												return false;
											}
										}
									}

									return true;
						}
					}
					if (buf.index >= no_buffers) {
						log[error] << "buf.index >= n_buffers!!!!" << std::endl;
						return false;
					}
					//if (!out[0].get()) return true;
					log[verbose_debug] << "Pushing frame with " << buf.bytesused
							<< "bytes" << std::endl;
					prepare_frame(reinterpret_cast<yuri::ubyte_t*>(buffers[buf.index].start),buf.bytesused);
					if (xioctl (fd, VIDIOC_QBUF, &buf)==-1) {
							log[error] << "VIDIOC_QBUF failed" << std::endl;
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
									log[error] << "VIDIOC_DQBUF failed" << std::endl;
									return false;
							}
					}
					prepare_frame(reinterpret_cast<yuri::ubyte_t*>(buf.m.userptr), buf.length);
					if (xioctl (fd, VIDIOC_QBUF, &buf)==-1) {
							log[error] << "VIDIOC_QBUF failed" << std::endl;
							return false;
					}
					break;
		case METHOD_NONE:
		default: return false;
	}
	if (output_frame && !buffer_free) {
		output_frame->set_time(0,0,frame_duration);
		if (out[0]) push_raw_video_frame(0,timestamp_frame(output_frame));
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
				log[error] << "VIDIOC_STREAMOFF failed" << std::endl;
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

yuri::uint_t V4l2Source::yuri_format_to_v4l2(yuri::format_t fmt)  throw (Exception)
{
	if (formats_map.count(fmt)) return formats_map[fmt];
	throw Exception("Unknown format");
}

yuri::format_t V4l2Source::v4l2_format_to_yuri(yuri::uint_t fmt)  throw (Exception)
{
	std::pair<yuri::format_t, yuri::uint_t> f;
	BOOST_FOREACH(f,formats_map) {
		if (f.second==fmt) return f.first;
	}
	//	case V4L2_PIX_FMT_SN9C20X_I420:	return YURI_FMT_YUV420_PLANAR;
	throw Exception("Unknown format");
}

bool V4l2Source::set_param(Parameter &param)
{
	if (param.name == "path") {
		filename = param.get<std::string>();
	} else if (param.name == "format") {
	std::string format = param.get<std::string>();
		yuri::format_t fmt = BasicPipe::get_format_from_string(format,YURI_TYPE_VIDEO);
		if (fmt && formats_map.count(fmt)) {
			pixelformat = formats_map[fmt];
		} else {
			// Process special formats....
//			else if (boost::iequals(format,"S920")) pixelformat = V4L2_PIX_FMT_SN9C20X_I420;
//			else if (boost::iequals(format,"BA81")) pixelformat = V4L2_PIX_FMT_SBGGR8;
			log[warning] << "Unsupported format specified. Trying YUV422"<<std::endl;
			pixelformat = V4L2_PIX_FMT_YUYV;
		}

	} else if (param.name == "width") {
		width = param.get<yuri::size_t>();
	} else if (param.name == "height") {
		height = param.get<yuri::size_t>();
	} else if (param.name == "method") {
	std::string method_s;
		method_s = param.get<std::string>();
		if (boost::iequals(method_s,"user")) method = METHOD_USER;
		else if (boost::iequals(method_s,"mmap")) method = METHOD_MMAP;
		else if (boost::iequals(method_s,"read")) method = METHOD_READ;
		else method=METHOD_NONE;
	} else if (param.name == "input") {
		input_number = param.get<yuri::ushort_t>();
	} else if (param.name == "illumination") {
			illumination = param.get<bool>();
	} else if (param.name == "combining") {
		combine_frames = param.get<bool>();
	} else if (param.name == "fps") {
		fps= param.get<yuri::size_t>();
	} else return BasicIOThread::set_param(param);
	return true;

}
bool V4l2Source::prepare_frame(yuri::ubyte_t *data, yuri::size_t size)
{
	//pBasicFrame  frame;
	yuri::format_t fmt = v4l2_format_to_yuri(pixelformat);
	if (!fmt) return false;
	FormatInfo_t fi = BasicPipe::get_format_info(fmt);
	if (!fi) return false;
	yuri::size_t frame_size = (width*height*fi->bpp)>>3;
	if (!fi->compressed) {
		if (!output_frame) {
			output_frame = allocate_empty_frame(fmt,width,height);
			buffer_free = frame_size;
		}
		yuri::size_t frame_position = frame_size - buffer_free;
		log[verbose_debug] << "Allocating " << fi->planes << " (got " << output_frame->get_planes_count() << ")" << std::endl;
		log[verbose_debug] << "Frame " << width << ", " << height << ", size: " << size<< std::endl;
		if (fi->planes==1) {
//			assert((*frame)[0].get_size()>=size);
			//yuri::size_t cp = size;
			if (size>buffer_free) size = buffer_free;
			memcpy(PLANE_RAW_DATA(output_frame,0)+frame_position,data,size);
			buffer_free -= size;
		} else {
			yuri::size_t offset = 0;
			for (yuri::size_t i = 0; i < fi->planes; ++i) {
				if (!size) break;
				yuri::size_t cols = width / fi->plane_x_subs[i];
				yuri::size_t rows = height / fi->plane_y_subs[i];
				yuri::size_t plane_size = (cols*rows*fi->component_depths[i])>>3;
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
				log[info] << "Copying " << plane_size << " bytes, have " << size-offset <<", free buffer: " << buffer_free<< std::endl;
				memcpy(PLANE_RAW_DATA(output_frame,i),data+offset,plane_size);
				offset+=plane_size;
				buffer_free-=plane_size;
			}
		}
	} else {
		output_frame = allocate_frame_from_memory(data,size);
		output_frame->set_parameters(fmt,width,height);
		buffer_free = 0;
	}
	// If we're no combining frames, we have to discard incomplete ones
	if (buffer_free && !combine_frames) {
		buffer_free = 0;
		log[warning] << "Discarding incomplete frame (missing " << buffer_free << " bytes)" << std::endl;
		output_frame.reset();
	}
	return true;
}

bool V4l2Source::open_file() throw (Exception)
{

	if (filename.empty()) throw Exception("Path must be specified!");
	//try {
	fd=open(filename.c_str(),O_RDWR|O_NONBLOCK);
	if (fd<0) {
		log[error] << "Failed to open file " << filename << std::endl;
		throw Exception("Failed to open file "+filename);
	}
	log[info] << filename << " opened successfully" << std::endl;
	//}
//	catch (boost::sy)

	return true;

}
bool V4l2Source::query_capabilities() throw(Exception)
{
	if (xioctl(fd,VIDIOC_QUERYCAP,&cap)<0) {
		log[error] << "VIDIOC_QUERYCAP ioctl failed!" << std::endl;
		throw Exception("VIDIOC_QUERYCAP ioctl failed!");
	}
	log[info]<< "Using driver: " << cap.driver << ", version " << ((cap.version >> 16) & 0xFF) << "." << ((cap.version >> 8) & 0xFF )<< "." <<  (cap.version & 0xFF) << std::endl;
	log[info]<< "Card name: " << cap.card << ", connected to: " << cap.bus_info << std::endl;
	if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE ) log[info] << "Device supports Video capture" << std::endl;
	else {
		log[error] << "Device does not supports Video capture!" << std::endl;
		throw Exception("Device does not support video capture!");
	}
	return true;
}
bool V4l2Source::enum_inputs() throw(Exception)
{
	v4l2_input input_info;
	input_info.index=0;
	while (xioctl(fd,VIDIOC_ENUMINPUT,&input_info)) {
		log[info] << "Input " << input_info.index << ": " << input_info.name <<
				", type: " << (input_info.type==V4L2_INPUT_TYPE_CAMERA?"camera":"tuner")
				<< ", status: " << (!input_info.status?"OK":(input_info.status==V4L2_IN_ST_NO_POWER?"No power":(input_info.status==V4L2_IN_ST_NO_SIGNAL?"No signal":"No color")))
				<< std::endl;
		input_info.index++;
	}
	number_of_inputs = input_info.index;
	return true;
}
bool V4l2Source::set_input() throw(Exception)
{
	// Not checking support for input 0 - many webcams does not report any input even thoug they have input 0
	if (input_number && input_number >= number_of_inputs) {
		// User is trying to set input not supported by the device. Let's only warn him here, the error will be returned later
		log[warning] << "The device reports that it does not support requested input "
				<< input_number << ". Trying to set it anyway" << std::endl;
	}
	log[debug] << "Setting input to " << input_number << std::endl;
	int inp = input_number;
	if (!xioctl (fd, VIDIOC_S_INPUT, &inp)) {
		log[debug] <<"VIDIOC_S_INPUT failed, input was NOT set. " << std::endl;
		// Let's assume that default input is 0. So not being able to set 0 is not really an error
		if (input_number) throw Exception("Failed to set input ");
	} else {
		log[info] << "Input set to " << input_number << std::endl;
	}
	return true;

}
bool V4l2Source::set_cropping() throw(Exception)
{
	v4l2_cropcap cropcap;
	v4l2_crop crop;
	memset(&cropcap,0,sizeof(cropcap));
	cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (!xioctl (fd, VIDIOC_CROPCAP, &cropcap)) {
		crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		crop.c = cropcap.defrect; /* reset to default */
		log[info] << "Selected input have pixel with aspect ratio " <<
				cropcap.pixelaspect.numerator << "/" << cropcap.pixelaspect.denominator << std::endl;
		if (xioctl (fd, VIDIOC_S_CROP, &crop) == -1) {
			log[warning] <<"VIDIOC_S_CROP failed, ignoring :)" << std::endl;
		}
	} else {
		log[warning] << "Failed to query cropping info, ignoring" << std::endl;
	}
	return true;
}
bool V4l2Source::enum_formats() throw (Exception)
{
	v4l2_fmtdesc fmts;
	supported_formats.clear();
	fmts.index=0;
	fmts.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
	while (!xioctl(fd,VIDIOC_ENUM_FMT,&fmts)) {
		log[info] << "Supported format " << fmts.index << ": " << fmts.description << std::endl;
		fmts.index++;
		supported_formats.push_back(fmts.pixelformat);
	}
	return true;
}
bool V4l2Source::set_format() throw (Exception)
{
	memset (&fmt,0,sizeof(v4l2_format));
	fmt.type=V4L2_BUF_TYPE_VIDEO_CAPTURE;
	if (xioctl(fd,VIDIOC_G_FMT,&fmt)<0) {
		log[error] << "VIDIOC_G_FMT ioctl failed! (" << strerror(errno)
							<< ")" << std::endl;
		throw Exception("Failed to get default format info!");
	}
	fmt.fmt.pix.pixelformat=pixelformat;
	fmt.fmt.pix.width=width;
	fmt.fmt.pix.height=height;
	if (xioctl(fd,VIDIOC_S_FMT,&fmt)<0) {
		log[error] << "VIDIOC_S_FMT ioctl failed!" << std::endl;
		throw Exception ("Failed to set input format!");
	}
	if (xioctl(fd,VIDIOC_G_FMT,&fmt)<0) {
		log[warning] << "Failed to verify if input format was set correctly !" << std::endl;
	}
	if (fmt.fmt.pix.pixelformat != pixelformat) {
		log[error] << "Failed to set requested input format!" << std::endl;
		throw Exception("Failed to set input format");
	}
	log[info] << "Video dimensions: " << fmt.fmt.pix.width << "x" << fmt.fmt.pix.height << std::endl;
	log[info] << "Pixel format (" << fmt.fmt.pix.pixelformat << "): " <<
		(char)(fmt.fmt.pix.pixelformat & 0xFF) <<
		(char)(fmt.fmt.pix.pixelformat>>8 & 0xFF) <<
		(char)(fmt.fmt.pix.pixelformat>>16 & 0xFF) <<
		(char)(fmt.fmt.pix.pixelformat>>24 & 0xFF)
		<< std::endl;
	log[info] << "Colorspace " << fmt.fmt.pix.colorspace << std::endl;
	imagesize=fmt.fmt.pix.sizeimage;
	width=fmt.fmt.pix.width;
	height=fmt.fmt.pix.height;
	pixelformat=fmt.fmt.pix.pixelformat;

	log[info] << "Image size: " << imagesize << std::endl;
	return true;
}
bool V4l2Source::enum_frame_intervals() throw (Exception)
{
	v4l2_frmivalenum frmvalen;
	frmvalen.pixel_format = pixelformat;
	frmvalen.width = width;
	frmvalen.height = height;
	frmvalen.index = 0;
	while (!xioctl(fd,VIDIOC_ENUM_FRAMEINTERVALS,&frmvalen)) {
		switch (frmvalen.type) {
		case V4L2_FRMIVAL_TYPE_CONTINUOUS:
			log[info] << "Supports continuous frame_intervals from"
				<< frmvalen.stepwise.min.numerator << "/" << frmvalen.stepwise.min.denominator
				<< "s to " << frmvalen.stepwise.max.numerator << "/" << frmvalen.stepwise.max.denominator <<"s"<< std::endl;
			break;
		case V4L2_FRMIVAL_TYPE_STEPWISE:
			log[info] << "Supports stepwise frame_intervals from"
				<< frmvalen.stepwise.min.numerator << "/" << frmvalen.stepwise.min.denominator
				<< "s to " << frmvalen.stepwise.max.numerator << "/" << frmvalen.stepwise.max.denominator
				<< "s with step " << frmvalen.stepwise.step.numerator << "/" << frmvalen.stepwise.step.denominator<<"s"<< std::endl;
			break;
		case V4L2_FRMIVAL_TYPE_DISCRETE:
			if (!frmvalen.index) log[info] << "Supports discrete frame_intervals:" << std::endl;
			log[info] << "\t"<<frmvalen.index<<": "<< frmvalen.discrete.numerator << "/" << frmvalen.discrete.denominator << "s" << std::endl;
			frmvalen.index++;
			break;
		}
		if (!frmvalen.index) break;
		//log[info] << "Supported frame_interval " << fmts.index << ": " << fmts.description << std::endl;

		//supported_formats.push_back(fmts.pixelformat);
	}
	return true;
}
bool V4l2Source::set_frame_params() throw (Exception)
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
		log[error] << "Failed to set frame parameters (FPS)" << std::endl;
		//throw Exception ("Failed to set input format!");
		return false;
	}
	if (xioctl(fd,VIDIOC_G_PARM,&strp)<0) {
		log[error] << "Failed to verify frame parameters (FPS)" << std::endl;
		//throw Exception ("Failed to set input format!");
		return false;
	}
	log[info] << "Driver reports current frame interval " << strp.parm.capture.timeperframe.numerator << "/"
			<< strp.parm.capture.timeperframe.denominator << "s" << std::endl;
	frame_duration = 1e6*strp.parm.capture.timeperframe.numerator/strp.parm.capture.timeperframe.denominator;
	return true;
}
bool V4l2Source::initialize_capture() throw (Exception)
{
	if (cap.capabilities & V4L2_CAP_STREAMING) {
		log[debug] << "Driver supports streaming operations, trying to initialize" << std::endl;
		if ((method==METHOD_NONE || method == METHOD_MMAP) && init_mmap()) {
			log[info] << "Initialized mmap " << std::endl;
			method=METHOD_MMAP;
		} else {
			log[debug] << "mmap failed, trying user pointers" << std::endl;
			if ((method==METHOD_NONE || method == METHOD_USER) && init_user()) {
				log[info] << "Initialized capture using user pointers" << std::endl;
				method=METHOD_USER;
			} else {
				log[debug] << "user pointers failed." << std::endl;
				method=METHOD_NONE;
			}
		}
	}
	else {
		log[debug] << "Driver does not support streaming operations!" << std::endl;
	}
	if(cap.capabilities & V4L2_CAP_READWRITE) {
		log[debug] << "Driver supports read/write operations" << std::endl;
		if (method==METHOD_NONE) {
			if (init_read()) {
				log[info] << "Initialized direct reading from device file" <<std::endl;
				method=METHOD_READ;
			}
		}
	}
	else {
		log[debug] << "Driver does not support read/write operations!" << std::endl;
	}
	if (method==METHOD_NONE) {
		log[fatal] << "I do not know how to read from this camera!" << std::endl;
		throw Exception("I do not know how to read from this camera!!");
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
bool V4l2Source::enable_iluminator() throw(Exception)
{
	if (illumination) {
		struct v4l2_queryctrl queryctrl;
		struct v4l2_control control;
		memset (&queryctrl, 0, sizeof (queryctrl));
		queryctrl.id=V4L2_CID_ILLUMINATORS_1;

		if (xioctl (fd, VIDIOC_QUERYCTRL, &queryctrl)<0) {
		        log[error] << "Illuminator is not supported" << std::endl;
		} else if (queryctrl.flags & V4L2_CTRL_FLAG_DISABLED) {
			log[error] << "Illuminator is disabled" << std::endl;
		} else {
			log[debug] << "Trying to enable illuminator " << queryctrl.name << std::endl;
			memset (&control, 0, sizeof (control));
			control.id = V4L2_CID_ILLUMINATORS_1;
			control.value = queryctrl.maximum;
			if (xioctl (fd, VIDIOC_S_CTRL, &control)<0) {
				log[error]<< "Failed to enable illuminator" << std::endl;
			} else {
				control.value = 0;
				if (xioctl (fd, VIDIOC_G_CTRL, &control)>=0) {
					if (control.value) {
						log[info] << "Illuminator enabled." <<std::endl;
					} else {
						log[error] << "Illuminator set, but camera reports it's not..." << std::endl;
					}
				} else {
					log[info]<<"Illuminator enabled, but failed to query it's status."<<std::endl;
				}
			}
		}
	}
	return true;
}

}
}

