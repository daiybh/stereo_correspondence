/*!
 * @file 		JPEGEncoder.cpp
 * @author 		Zdenek Travnicek
 * @date 		29.7.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "JPEGEncoder.h"

namespace yuri {

namespace io {

REGISTER("jpegencoder",JPEGEncoder)

shared_ptr<BasicIOThread> JPEGEncoder::generate(Log &_log,pThreadBase parent,Parameters& parameters) throw (Exception)
{
	shared_ptr<BasicIOThread> jpg (new JPEGEncoder(_log,parent,
			parameters["level"].get<int>(),
			parameters["buffer"].get<long>()));
	return jpg;
}
shared_ptr<Parameters> JPEGEncoder::configure()
{
	shared_ptr<Parameters> p (new Parameters());
	(*p)["level"]["Compression level"]=75;
	(*p)["buffer"]["Buffer size. Default value should work for most situations."]=1048576;
	p->set_max_pipes(1,1);
	p->add_input_format(YURI_FMT_RGB);
	p->add_output_format(YURI_IMAGE_JPEG);
	p->add_converter(YURI_FMT_RGB,YURI_IMAGE_JPEG,false);
	return p;
}

//bool JPEGEncoder::configure_converter(Parameters& parameters,long format_in,
//		long format_out)	throw (Exception)
//{
//	if (format_out != YURI_IMAGE_JPEG) throw NotImplemented();
//	if (format_in != YURI_FMT_RGB) throw NotImplemented();
//	return true;
//}

JPEGEncoder::JPEGEncoder(Log &_log, pThreadBase parent,int level,
		long buffer_size)
	:BasicIOThread(_log,parent,1,1,"JPG Enc"),level(level),
	buffer_size(buffer_size),width(0),height(0)
{
}

JPEGEncoder::~JPEGEncoder()
{
}

bool JPEGEncoder::step()
{
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;
	log[debug] << "Reading packet " << frame->get_size() << " bytes long" << std::endl;


	//int bpp = colorspace==YURI_COLORSPACE_RGB?3:4;
	boost::posix_time::ptime t1(boost::posix_time::microsec_clock::universal_time());
	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr       jerr;

	width = frame->get_width();
	height = frame->get_height();
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_compress(&cinfo);
	setDestManager(&cinfo);
	cinfo.image_width      = width;
	cinfo.image_height     = height;
	yuri::size_t Bpp;
	switch (frame->get_format()) {
		case YURI_FMT_RGB:
			cinfo.input_components = 3;
			cinfo.in_color_space   = JCS_RGB;
			jpeg_set_defaults(&cinfo);
			Bpp=3;
			break;
			// TODO: implement YUV input for JPEGEncoder
		/*case YURI_FMT_YUV422:
			cinfo.input_components = 3;
			cinfo.in_color_space   = JCS_YCbCr;
			jpeg_set_defaults(&cinfo);
			cinfo.comp_info[0].h_samp_factor = 2;
			cinfo.comp_info[0].v_samp_factor = 1;
			cinfo.comp_info[1].h_samp_factor = 1;
			cinfo.comp_info[1].v_samp_factor = 1;
			cinfo.comp_info[2].h_samp_factor = 1;
			cinfo.comp_info[2].v_samp_factor = 1;
			cinfo.comp_info[1].component_index=2;
			cinfo.comp_info[2].component_index=1;
			Bpp=2;
			break;*/
		default:
			log[error] << "Unsupported input format " << BasicPipe::get_format_string(frame->get_format())<< std::endl;
			return true;
	}



	/*set the quality [0..100]  */
	jpeg_set_quality (&cinfo, level, true);
	jpeg_start_compress(&cinfo, true);
	JSAMPROW row_pointer;

	while (cinfo.next_scanline < cinfo.image_height) {
		row_pointer = reinterpret_cast<JSAMPROW>(PLANE_RAW_DATA(frame,0)+cinfo.next_scanline*width*Bpp );
		jpeg_write_scanlines(&cinfo, &row_pointer, 1);
	}
	jpeg_finish_compress(&cinfo);
	boost::posix_time::ptime t2(boost::posix_time::microsec_clock::universal_time());
		boost::posix_time::time_period tp(t1,t2);
		log[debug] << "JPEG compression took: " << tp.length().total_microseconds()
			<< " us" << std::endl;
	dumpData();
    jpeg_destroy_compress(&cinfo);
	return true;
}

void JPEGEncoder::setDestManager(jpeg_compress_struct* cinfo)
{
	cinfo->dest = new jpeg_destination_mgr;
	cinfo->dest->init_destination=sInitDestination;
	cinfo->dest->empty_output_buffer=sEmptyBuffer;
	cinfo->dest->term_destination=sTermDestination;
	cinfo->client_data=(void*)this;
}

void JPEGEncoder::initDestination(j_compress_ptr cinfo)
{
	log[verbose_debug] << "Initializin dest" << std::endl;
	temp_data.seekg(0,std::ios::beg);
	temp_data.seekp(0,std::ios::beg);
	temp_data.str().clear();
	if (!buffer) buffer.reset(new char[buffer_size]);
	cinfo->dest->next_output_byte=(JOCTET *) buffer.get();
	cinfo->dest->free_in_buffer=buffer_size;
}
int JPEGEncoder::emptyBuffer(j_compress_ptr cinfo)
{
	log[verbose_debug] << "flushing " << (buffer_size - cinfo->dest->free_in_buffer)
		<< " bytes" << std::endl;
	temp_data.write((const char*)buffer.get(),
			(buffer_size - cinfo->dest->free_in_buffer));
	cinfo->dest->free_in_buffer=buffer_size;
	return TRUE;
}

void JPEGEncoder::sInitDestination(j_compress_ptr cinfo)
{
	JPEGEncoder *jpg = (JPEGEncoder*)cinfo->client_data;
	jpg->initDestination(cinfo);
}
int JPEGEncoder::sEmptyBuffer(j_compress_ptr cinfo)
{
	JPEGEncoder *jpg = (JPEGEncoder*)cinfo->client_data;
	return jpg->emptyBuffer(cinfo);
}
void JPEGEncoder::sTermDestination(j_compress_ptr cinfo)
{
	JPEGEncoder *jpg = (JPEGEncoder*)cinfo->client_data;
	jpg->emptyBuffer(cinfo);
}

yuri::size_t JPEGEncoder::dumpData()
{
	yuri::size_t length = temp_data.tellp();
	log[verbose_debug] << "Reading " << length << " bytes fromstd::stringstream" << std::endl;
	temp_data.seekg(0,std::ios::beg);
	if (out[0]) {
		const std::string& str = temp_data.str();
		pBasicFrame f = allocate_frame_from_memory(reinterpret_cast<const yuri::ubyte_t*>(str.data()), str.size());
		push_video_frame(0,f,width,height,YURI_IMAGE_JPEG);
	} else {
		length = 0;
	}
	temp_data.seekp(0,std::ios::beg);
	temp_data.str().clear();
	return length;
//	shared_array<yuri::ubyte_t> mem = allocate_memory_block(length);

//	temp_data.read(reinterpret_cast<char*>(mem.get()),length);
//	temp_data.seekp(0,std::ios::beg);
//	temp_data.str().clear();
//	if (!out[0]) {
//		return 0;
//	}
//	else {
//		pBasicFrame f = allocate_frame_from_memory(mem, length);
//		push_video_frame(0,f,width,height,YURI_IMAGE_JPEG);
//	}
//	return length;
}




}



}
