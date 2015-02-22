/*!
 * @file 		PdfSource.cpp
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#include "PdfSource.h"
#include "yuri/core/Module.h"
#include "yuri/core/frame/RawVideoFrame.h"
#include "yuri/core/frame/raw_frame_types.h"
#include "yuri/core/utils/assign_events.h"
#include <poppler-page.h>
#include <poppler-page-renderer.h>

#include <map>
namespace yuri {
namespace pdf_source {


IOTHREAD_GENERATOR(PdfSource)

MODULE_REGISTRATION_BEGIN("pdf_source")
		REGISTER_IOTHREAD("pdf_source",PdfSource)
MODULE_REGISTRATION_END()

core::Parameters PdfSource::configure()
{
	core::Parameters p = core::IOThread::configure();
	p.set_description("PdfSource");
	p["filename"]["PDF filename"]="";
	p["page"]["Starting page"]=0;
	p["resolution"]["Output resolution. Ignored when DPI is specified"]=resolution_t{800,600};
	p["dpi"]["DPI for rendering. If >0, then resolution is ignored"]=0.0;

	return p;
}


PdfSource::PdfSource(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters):
core::IOThread(log_,parent,1,1,std::string("pdf_source")),
BasicEventConsumer(log),
BasicEventProducer(log),
page_(0),resolution_{800,600},dpi_(0.0),changed_(true),page_count_(0)
{
	IOTHREAD_INIT(parameters)
	document_.reset(poppler::document::load_from_file(filename_.c_str()));
	if (!document_) throw exception::InitializationFailed("Failed to load " + filename_);
	int major = 0, minor = 0;
	document_->get_pdf_version(&major, &minor);
	page_count_ = document_->pages();
	log[log::info] << filename_ << " loaded PDF " << major << "." << minor << " with " << page_count_ << " pages.";

}

PdfSource::~PdfSource() noexcept
{
}

namespace {

const std::map<poppler::image::format_enum, format_t> poppler_formats = {
		{poppler::image::format_mono, core::raw_format::y8},
		{poppler::image::format_rgb24, core::raw_format::bgr24},
		{poppler::image::format_argb32, core::raw_format::bgra32}
};

format_t get_poppler_fmt(poppler::image::format_enum fmt)
{
	auto it = poppler_formats.find(fmt);
	if (it == poppler_formats.end()) return 0;
	return it->second;
}
core::pFrame get_frame(const poppler::image& image)
{
	const auto fmt = get_poppler_fmt(image.format());
	if (!fmt) return {};
	resolution_t res {static_cast<dimension_t>(image.width()), static_cast<dimension_t>(image.height())};
	// Ignoring line width - this may cause problems...
	return core::RawVideoFrame::create_empty(fmt, res, reinterpret_cast<const uint8_t*>(image.const_data()), image.bytes_per_row() * image.height());
}

constexpr const double default_dpi = 72.0;
}

void PdfSource::run()
{
	emit_event("total", page_count_);
	emit_event("max", page_count_);
	emit_event("last", page_count_-1);
	while(still_running()) {
		wait_for_events(get_latency());
		process_events();
		if (changed_) {
			std::unique_ptr<poppler::page> page (document_->create_page(page_));
			poppler::page_renderer renderer;
			renderer.set_render_hints(poppler::page_renderer::antialiasing |
									poppler::page_renderer::text_antialiasing |
									poppler::page_renderer::text_hinting);
			double dpi = dpi_;
			emit_event("index", page_);
			emit_event("page", page_);
			if (dpi_ <= 0.0) {
				auto rect = page->page_rect();
				dpi = default_dpi * std::min(resolution_.width/static_cast<double>(rect.width()),
						resolution_.height/static_cast<double>(rect.height()));
				log[log::verbose_debug] << "Rendering using DPI " << dpi;
			}

			auto image = renderer.render_page(page.get(), dpi, dpi);
			auto frame = get_frame(image);

			push_frame(0, frame);
			changed_ = false;
		}
	}
}

bool PdfSource::set_param(const core::Parameter& param)
{
	if (assign_parameters(param)
			(filename_, "filename")
			(page_, "page")
			(resolution_, "resolution")
			(dpi_, "dpi"))
		return true;
	return core::IOThread::set_param(param);
}

bool PdfSource::do_process_event(const std::string& event_name, const event::pBasicEvent& event)
{
	if (assign_events(event_name, event)
			(page_, "page", "index")
			.bang("next", [this](){++page_;})
			.bang("prev", [this](){if (page_>0)--page_;})
			.bang("reload", [](){})) {
		page_ = std::min(page_count_-1, std::max(0,page_));
		changed_ = true;
		return true;
	}
	return false;
}

} /* namespace pdf_source */
} /* namespace yuri */
