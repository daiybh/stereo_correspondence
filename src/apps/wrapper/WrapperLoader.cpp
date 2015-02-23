/*
 * WrapperLoader.cpp
 *
 *  Created on: 24. 1. 2015
 *      Author: neneko
 */

#include "WrapperLoader.h"

#include <sys/types.h>
#include <signal.h>
#include <unistd.h>

namespace yuri{
namespace wrapper {


WrapperLoader::WrapperLoader(log::Log& log_, const std::string& config_file, const std::string& node_name)
		:IOThread(log_, {}, 0,0,"wrap_builder"),config_file_(config_file),
		 node_name_(node_name)
	{
		log[log::info] << "wrap builder";
		init();

	}
	WrapperLoader::~WrapperLoader() noexcept
	{

	}
	void* WrapperLoader::get_func(const std::string& name) {
		auto g = grabber_;
		if (g) return g->get_func(name);
		return nullptr;
	}

	void WrapperLoader::init()
	{
		// We need to make sure the builder is not initialized twice
		lock_t _(builder_mutex_);
		if (!graph_) {
			log[log::info] << "Preparing builder";
			graph_ = std::make_shared<xml_wrapper>(log, *this, config_file_);
			spawn_thread(graph_);

		}


	}
	void WrapperLoader::set_viewport(geometry_t geometry)
	{
		auto g = get_grabber();
		if (g) g->set_viewport(geometry);
	}


	void WrapperLoader::pre_swap()
	{
		auto g = get_grabber();
		if (g) g->pre_swap();

	}
	void WrapperLoader::post_swap()
	{
		auto g = get_grabber();
		if (g) g->post_swap();

	}
	std::shared_ptr<fb_grabber::FBGrabber> WrapperLoader::get_grabber()
	{
		if (grabber_) return grabber_;

		lock_t _(builder_mutex_);
		grabber_ = std::dynamic_pointer_cast<fb_grabber::FBGrabber>(graph_->get_node(node_name_));

		if (grabber_) {
			log[log::info] << "Got grabber";
		}
		return grabber_;

	}

	void WrapperLoader::child_ends_hook(core::pwThreadBase /* child */, int /* code */,
						size_t /* remaining_child_count */)
	{
		log[log::info] << "End";
		if (::kill(::getpid(), SIGTERM)) {
			log[log::warning] << "Failed to send kill signal";
		}
	}
}
}
