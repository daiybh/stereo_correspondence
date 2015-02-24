/*!
 * @file 		PdfSource.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		22.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef PDFSOURCE_H_
#define PDFSOURCE_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"
#include <poppler-document.h>

namespace yuri {
namespace pdf_source {

enum class caching_mode_t: uint8_t;

class PdfSource: public core::IOThread,
				public event::BasicEventConsumer,
				public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	PdfSource(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~PdfSource() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;

	core::pFrame get_frame_from_index(int index);
	core::pFrame get_frame_from_cache(int index);
private:
	std::string filename_;
	std::unique_ptr<poppler::document> document_;
	int page_;
	resolution_t resolution_;
	double dpi_;
	bool changed_;
	int page_count_;
	caching_mode_t caching_mode_;
//	size_t cache_size_;

	std::map<int, core::pFrame> frame_cache_;
};

} /* namespace pdf_source */
} /* namespace yuri */
#endif /* PDFSOURCE_H_ */

