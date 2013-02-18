/*!
 * @file 		DXTCompress.cpp
 * @author 		Zdenek Travnicek
 * @date 		11.2.2013
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2013
 * 				Distributed under GNU Public License 3.0
 *
 */

#include "DXTCompress.h"
#include "yuri/config/RegisteredClass.h"
#include "squish.h"
namespace yuri {
namespace dummy_module {

REGISTER("dxt_compress",DXTCompress)

IO_THREAD_GENERATOR(DXTCompress)

using namespace yuri::io;
using boost::iequals;
shared_ptr<Parameters> DXTCompress::configure()
{
	shared_ptr<Parameters> p = BasicIOThread::configure();
	p->set_description("Dummy module. For testing only.");
	(*p)["type"]["Set type of compression (DXT1, DXT3, DXT5)"]="DXT1";
	p->set_max_pipes(1,1);
	return p;
}

DXTCompress::DXTCompress(Log &log_,pThreadBase parent,Parameters &parameters):
BasicIOThread(log_,parent,1,1,std::string("dummy")),dxt_type(squish::kDxt1),
format(YURI_FMT_DXT1)
{
	IO_THREAD_INIT("Dummy")
}

DXTCompress::~DXTCompress()
{
}

bool DXTCompress::step()
{
	pBasicFrame frame = in[0]->pop_frame();
	if (frame) {
		yuri::format_t fmt = frame->get_format();
		if (fmt!=YURI_FMT_RGBA) {
			log[warning] << "Unsupported format " << BasicPipe::get_format_string(fmt)
				<< ". Only RGBA (RGB32) supported\n";
			return true;
		}
		yuri::size_t width  = frame->get_width();
		yuri::size_t height = frame->get_height();
		if ((width % 4) || (height % 4)) {
			// The library can accept even weird size images, so we silently ignore it
			log[debug] << "Input image has to be divisible by 4 in both directions ("
					<< width << "x" << height << "\n";
		}
		log[verbose_debug] << "Converting a frame " << width << "x" << height << "\n";
		const yuri::size_t image_size = squish::GetStorageRequirements(width,height,dxt_type);
		pBasicFrame output(new BasicFrame(1));
		PLANE_DATA(output,0).resize(image_size);
		squish::CompressImage(PLANE_RAW_DATA(frame,0),width,height,PLANE_RAW_DATA(output,0),dxt_type|squish::kColourRangeFit);
		if (width&3) width=(width&~3)+4;
		if (height&3) height=(height&~3)+4;
		push_video_frame(0, output, format, width, height);
		log[verbose_debug] << "Frame pushed, " << output->get_width() << "x" << output->get_height() << ", size: " << image_size <<" bytes\n";
	}
	return true;
}
bool DXTCompress::set_param(Parameter& param)
{
	if (param.name == "type") {
		if (iequals(param.get<std::string>(),"DXT5")) {
			dxt_type = squish::kDxt5; format = YURI_FMT_DXT5;
			log[info] << "Using format DXT5\n";
		} else if (iequals(param.get<std::string>(),"DXT3")) {
			dxt_type = squish::kDxt3; format = YURI_FMT_DXT3;
			log[info] << "Using format DXT3\n";
		} else {
			dxt_type = squish::kDxt1; format = YURI_FMT_DXT1;
			log[info] << "Using format DXT1\n";
		}
	} else return BasicIOThread::set_param(param);
	return true;
}

} /* namespace dummy_module */
} /* namespace yuri */
