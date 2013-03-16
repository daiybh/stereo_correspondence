/*!
 * @file 		JPEGDecoder.cpp
 * @author 		Zdenek Travnicek
 * @date 		3.8.2009
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2009 - 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "JPEGDecoder.h"
#include "yuri/core/Module.h"
#include <boost/assign.hpp>
namespace yuri {

namespace jpg {

REGISTER("jpegdecoder",JPEGDecoder)
IO_THREAD_GENERATOR(JPEGDecoder)


namespace {
	std::map<format_t, J_COLOR_SPACE> yuri_to_jpg_formats = boost::assign::map_list_of<format_t, J_COLOR_SPACE>
	(YURI_FMT_RGB,JCS_RGB)
	(YURI_FMT_YUV444,JCS_YCbCr);
}
//core::pBasicIOThread JPEGDecoder::generate(log::Log &_log,core::pwThreadBase parent, core::Parameters& parameters)
//{
//	shared_ptr<JPEGDecoder> j( new JPEGDecoder(_log,parent));
//	j->forceLineWidthMult(parameters["line_multiply"].get<int>());
//	return j;
//}
core::pParameters JPEGDecoder::configure()
{
	core::pParameters p = BasicIOThread::configure();
	(*p)["line_multiply"]=1;
	(*p)["format"]["Output format"]=std::string("YUV444");
	(*p)["fast"]["Fast decoding (slightly worse quality)"]=false;
	return p;
}


JPEGDecoder::JPEGDecoder(log::Log &_log, core::pwThreadBase parent, core::Parameters& parameters)
	:core::BasicIOThread(_log,parent,1,1,"JPEGDecoder"),line_width_mult(1),aborted(false),
	 format_(YURI_FMT_YUV444),raw_(false),fast_(false)
{
	IO_THREAD_INIT("[JPEGDecoder]");
}

JPEGDecoder::~JPEGDecoder() {
}

namespace {
format_t get_format(const jpeg_decompress_struct& cinfo)
{
	if (cinfo.comp_info[0].h_samp_factor ==2 && cinfo.comp_info[0].v_samp_factor==1 &&
			cinfo.comp_info[1].h_samp_factor == 1 && cinfo.comp_info[1].v_samp_factor==1 &&
			cinfo.comp_info[2].h_samp_factor == 1 && cinfo.comp_info[2].v_samp_factor==1) {
		return YURI_FMT_YUV420_PLANAR;
	}
	return YURI_FMT_NONE;
}
}
bool JPEGDecoder::step() {
	int width, height, colorspace;
	if (!in[0] || !(frame = in[0]->pop_frame()))
		return true;
	bool decompressed = false;
	aborted = false;
	//log[log::debug] << "Reading packet " << frame->get_size() << " bytes long" << std::endl;

	struct jpeg_decompress_struct cinfo;
	cinfo.client_data=reinterpret_cast<void*>(this);
	struct jpeg_error_mgr jerr;
	cinfo.err = jpeg_std_error(&jerr);
	cinfo.err->error_exit=errorExit;
	jpeg_create_decompress(&cinfo);
	setDestManager(&cinfo);


	int bpp;
	int linesize;
	shared_array<yuri::ubyte_t> mem;
	core::pBasicFrame out_frame;// (new core::BasicFrame(1));
	//while (cinfo.next_scanline < cinfo.image_height) {
	try {
		if (jpeg_read_header(&cinfo,true)!=JPEG_HEADER_OK) {
			log[log::warning] << "Unrecognized file header!!" << std::endl;
			return true;
		}
		if (!raw_) {
		//log[log::info] << "jpg colorspace: " << cinfo.jpeg_color_space << ", out csp: " << cinfo.out_color_space;
			cinfo.out_color_space = yuri_to_jpg_formats[format_];
			colorspace = format_;
			if (fast_) {
				cinfo.do_fancy_upsampling=false;
				cinfo.dct_method = JDCT_IFAST;
//				cinfo.dct_method = JDCT_FLOAT;
				cinfo.do_block_smoothing = false;
//				cinfo.quantize_colors = false;
			}
		} else {
			cinfo.out_color_space = cinfo.jpeg_color_space;
			cinfo.raw_data_out = true;
			colorspace = get_format(cinfo);
		}
		if (colorspace == YURI_FMT_NONE) aborted = true;
		jpeg_start_decompress(&cinfo);
//		log[log::info] << "jpg colorspace: " << cinfo.jpeg_color_space << ", out csp: " << cinfo.out_color_space<<", componenets: "<<cinfo.output_components;
		//for (int i=0;i<3;++i) log[log::info] << "comp " << i << ", x: " << cinfo.comp_info[i].h_samp_factor <<", y: " << cinfo.comp_info[i].v_samp_factor;
		if (aborted) throw (yuri::exception::Exception("Decoding failed"));
		width = cinfo.image_width;
		if (line_width_mult > 1 && (width % line_width_mult))
			width = width + line_width_mult - (width % line_width_mult);
		height = cinfo.image_height;
		bpp = cinfo.output_components;
		linesize = width*bpp;
		//colorspace= (bpp==3?YURI_FMT_RGB24:bpp==4?YURI_FMT_RGB32:YURI_FMT_NONE);
//		log[log::verbose_debug] << "Allocating " << linesize * height << " bytes for image "
//			<< width << "x" << height << " @ " << 8*bpp << "bpp" << std::endl;
//		PLANE_DATA(out_frame,0).resize(linesize * height);
		out_frame = allocate_empty_frame(colorspace,width,height);
//		mem=allocate_memory_block(linesize * height);
//		(*out_frame)[0].set(mem,linesize * height);


		int planes = raw_?3:1;
		std::vector<std::vector<JSAMPROW> > row_pointers(planes);
		for (int p=0;p<planes;++p) {
			row_pointers[p].resize(height);
			int ls = linesize*cinfo.comp_info[p].h_samp_factor/cinfo.comp_info[0].h_samp_factor;
			for (int h=0;h<height;++h) {
				row_pointers[p][h]=PLANE_RAW_DATA(out_frame,p) + h*ls;
			}
		}
		std::vector<JSAMPARRAY> arrays_pointers(planes);
		for (int p=0;p<planes;++p) {
			arrays_pointers[p]=&row_pointers[p][0];
		}

		yuri::size_t completed = 0, processed=0;
		//for (int i=0;i<height;++i) {
//		JSAMPARRAY &ptrs = row_pointers[0][0];
		while (cinfo.output_scanline < cinfo.image_height) {
//			row_pointer = (JSAMPROW) PLANE_RAW_DATA(out_frame,0) + i*linesize;
			if (!raw_) processed = jpeg_read_scanlines(&cinfo, &row_pointers[0][cinfo.output_scanline], height);
			else {
//				log[log::info] << "processing raw line " << cinfo.output_scanline;
				//JSAMPARRAY arr_pointer = &row_pointers[0];
				for (int p=0;p<planes;++p) {
					arrays_pointers[p]=&row_pointers[p][0]+cinfo.output_scanline;
				}
				JSAMPIMAGE img = &arrays_pointers[0];
				processed = jpeg_read_raw_data(&cinfo, img, height-cinfo.output_scanline);
			}
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
//			completed += processed;
//			if (static_cast<int>(completed) >= height) break;
			if (!processed) {
				log[log::error] << "No lines processed ... corrupt file?" << std::endl;
				break;
			}
		}
		if (processed) {
			jpeg_finish_decompress(&cinfo);
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
			decompressed = true;
			jpeg_destroy_decompress(&cinfo);
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
		} else {
			decompressed = false;
			jpeg_destroy_decompress(&cinfo);
			if (aborted) throw (yuri::exception::Exception("Decoding failed"));
		}
	}
	catch (yuri::exception::Exception &e) {
		log[log::error] << "Decoding failed!: " << e.what() << std::endl;
		jpeg_abort(reinterpret_cast<j_common_ptr>(&cinfo));
		return true;
	}


	if (decompressed && out[0] && out_frame) {
		push_video_frame(0,out_frame,colorspace,width,height);
		out_frame.reset();
		//out[0]->set_params(width,height,colorspace);
		//out[0]->push_frame(mem,linesize*height,true);
	}
	return true;
}

bool JPEGDecoder::set_param(const core::Parameter& param)
{
	if (param.name=="line_multiply") {
		int mult = param.get<int>();
		if (mult>1)line_width_mult = mult; else mult=1;
	} else if (param.name=="fast") {
		fast_ = param.get<bool>();
	} else if (param.name=="format") {
		std::string fmt = param.get<std::string>();
		if (fmt == "raw") {
			format_ = YURI_FMT_NONE;
			raw_ = true;
		} else {
			raw_ = false;
			format_ = core::BasicPipe::get_format_from_string(fmt);
			if (yuri_to_jpg_formats.count(format_)==0) {
				log[log::warning] << "Unsupported output format, using YUV444";
				format_ = YURI_FMT_YUV444;
			}
		}
	} else return core::BasicIOThread::set_param(param);
	return true;
}

void JPEGDecoder::setDestManager(jpeg_decompress_struct* cinfo)
{
	cinfo->src = new jpeg_source_mgr;
	cinfo->src->init_source=initSrc;
	cinfo->src->next_input_byte=(JOCTET *)PLANE_RAW_DATA(frame,0);
	cinfo->src->bytes_in_buffer=PLANE_SIZE(frame,0);
	cinfo->src->fill_input_buffer=fillInput;
	cinfo->src->resync_to_restart=resyncData;
	cinfo->src->skip_input_data=skipData;
	cinfo->src->term_source=termSource;

}

/// Check if there's valid JPEG magic
bool JPEGDecoder::validate(core::pBasicFrame frame)
{
	if (!frame || PLANE_SIZE(frame,0)<4) return false;
	uint8_t *magic = reinterpret_cast<uint8_t*>(PLANE_RAW_DATA(frame,0));
	if  (magic[0] == 0xff &&
		 magic[1] == 0xd8 ) return true;
		/*(magic[2] == 0xff) || (magic[2] == 0x00))
			return true;*/
//	cout << "Magic: " << hex << (uint)magic[0] << (uint)magic[1] << dec << std::endl;

	return false;
}

void JPEGDecoder::initSrc(jpeg_decompress_struct* /*cinfo*/)
{
}

int JPEGDecoder::fillInput(jpeg_decompress_struct* /*cinfo*/)
{
	return 1;
}
void JPEGDecoder::skipData(jpeg_decompress_struct* cinfo, long numbytes)
{
	if ((long)(cinfo->src->bytes_in_buffer) < numbytes) cinfo->src->bytes_in_buffer=0;
	else {
		cinfo->src->bytes_in_buffer-=numbytes;
		cinfo->src->next_input_byte+=numbytes;
	}
}
int JPEGDecoder::resyncData(jpeg_decompress_struct* /*cinfo*/, int /*desired*/)
{
// Guessing return value...
	return 1;
}
void JPEGDecoder::termSource(jpeg_decompress_struct* /*cinfo*/)
{
}

void JPEGDecoder::errorExit(jpeg_common_struct* cinfo)
{
	JPEGDecoder *dec = reinterpret_cast<JPEGDecoder*>(cinfo->client_data);
	if (dec) dec->abort();
}
void JPEGDecoder::abort()
{
	aborted =  true;
}
}

}
