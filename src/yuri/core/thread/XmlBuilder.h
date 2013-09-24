/*
 * XMLBuilder.h
 *
 *  Created on: 15.9.2013
 *      Author: neneko
 */

#ifndef XMLBUILDER_H_
#define XMLBUILDER_H_

#include "IOThread.h"
#include "yuri/event/BasicEventParser.h"
namespace yuri {
namespace core {

struct variable_info_t {
	std::string name;
	std::string description;
	std::string value;
};

class XmlBuilder: public IOThread, public event::BasicEventParser
{
public:
	static Parameters configure();
	IOTHREAD_GENERATOR_DECLARATION
	XmlBuilder(const log::Log& log_, pwThreadBase parent, const Parameters& parameters);
	XmlBuilder(const log::Log& log_, pwThreadBase parent, const std::string& filename, const std::vector<std::string>& argv, bool parse_only = false);
	~XmlBuilder() noexcept;

	const std::string& get_app_name();
	const std::string& get_description();
	std::vector<variable_info_t> get_variables() const;
private:
	virtual void run() override;
	virtual bool step() override;
	virtual bool set_param(const Parameter& parameter) override;
	virtual event::pBasicEventProducer find_producer(const std::string& name) override;
	virtual event::pBasicEventConsumer find_consumer(const std::string& name) override;
	virtual bool 				do_process_event(const std::string& event_name, const event::pBasicEvent& event) override;
	struct builder_pimpl_t;
	unique_ptr<builder_pimpl_t>	pimpl_;
	std::string	filename_;
};
}
}



#endif /* XMLBUILDER_H_ */
