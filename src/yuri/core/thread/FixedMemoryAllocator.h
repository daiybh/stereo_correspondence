/*!
 * @file 		FixedMemoryAllocator.h
 * @author 		Zdenek Travnicek
 * @date 		28.1.2012
 * @date		21.11.2013
 * @copyright	Institute of Intermedia, CTU in Prague, 2012 - 2013
 * 				Distributed under modified BSD Licence, details in file doc/LICENSE
 *
 * @details		FixedMemoryAllocator implements effective allocation of
 *  equally sized blocks of memory.
 *  It managed pool of allocated blocks and serves them to application,
 *  reclaiming unused blocks back to pool.
 *  It does NOT explicitly deallocate block unless asked to!!
 *  This could lead to potentially high memory consumption.
 */

#ifndef FIXEDMEMORYALLOCATOR_H_
#define FIXEDMEMORYALLOCATOR_H_

#include "yuri/core/thread/IOThread.h"
namespace yuri {

namespace core {


class FixedMemoryAllocator: public IOThread {
public:
	struct Deleter {
		Deleter(yuri::size_t size, uint8_t *original_pointer):
			size(size),original_pointer(original_pointer) {}
		Deleter(const Deleter& d)noexcept:size(d.size),original_pointer(d.original_pointer) {}
		void operator()(void *mem) const noexcept;
		/**\brief Size of block associated with this object */
		yuri::size_t size;
		/**\brief Pointer to the memory block associated with this object */
		uint8_t *original_pointer;

	};
	typedef std::pair<uint8_t*, struct Deleter> memory_block_t;
	IOTHREAD_GENERATOR_DECLARATION
	EXPORT static Parameters configure();
	EXPORT FixedMemoryAllocator(log::Log &_log, pwThreadBase parent, const Parameters &parameters);
	EXPORT virtual ~FixedMemoryAllocator() noexcept;
	EXPORT static memory_block_t get_block(yuri::size_t size);
	EXPORT static bool return_memory(yuri::size_t size, uint8_t* mem);
	EXPORT static bool allocate_blocks(yuri::size_t size, yuri::size_t count);
	EXPORT static bool remove_blocks(yuri::size_t size, yuri::size_t count=0);
	EXPORT static size_t preallocated_blocks(size_t size);
	EXPORT static std::pair<size_t, size_t> clear_all();
private:

	bool step();
	EXPORT virtual bool set_param(const Parameter &parameter);
	static bool do_allocate_blocks(yuri::size_t size, yuri::size_t count);

	/**\brief Global mutex protecting the pool */
	static mutex mem_lock;
	/**\brief Global memory pool */
	static std::map<yuri::size_t, std::vector<uint8_t* > > memory_pool;
	/**\brief Size of the blocks this object allocates */
	yuri::size_t block_size;
	/**\brief Number of the blocks this object allocates */
	yuri::size_t count;
};

}

}

#endif /* FIXEDMEMORYALLOCATOR_H_ */
