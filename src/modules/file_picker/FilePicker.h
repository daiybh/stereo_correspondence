/*!
 * @file 		FilePicker.h
 * @author 		Zdenek Travnicek <travnicek@iim.cz>
 * @date 		21.02.2015
 * @copyright	Institute of Intermedia, CTU in Prague, 2015
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 */

#ifndef FILEPICKER_H_
#define FILEPICKER_H_

#include "yuri/core/thread/IOThread.h"
#include "yuri/event/BasicEventConsumer.h"
#include "yuri/event/BasicEventProducer.h"
namespace yuri {
namespace file_picker {

struct pattern_detail_t {
	std::string head;
	std::string tail;
	index_t counter;
	bool fill;
};


class FilePicker: 	public core::IOThread,
					public event::BasicEventConsumer,
					public event::BasicEventProducer
{
public:
	IOTHREAD_GENERATOR_DECLARATION
	static core::Parameters configure();
	FilePicker(const log::Log &log_, core::pwThreadBase parent, const core::Parameters &parameters);
	virtual ~FilePicker() noexcept;
private:
	
	virtual void run() override;
	virtual bool set_param(const core::Parameter& param) override;
	virtual bool do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
private:

	std::string pattern_;
	index_t index_;
	double fps_;
	format_t format_;
	bool raw_format_;
	bool changed_;
	bool scan_total_;
	pattern_detail_t pattern_detail_;
	resolution_t resolution_;
};

} /* namespace file_picker */
} /* namespace yuri */
#endif /* FILEPICKER_H_ */
