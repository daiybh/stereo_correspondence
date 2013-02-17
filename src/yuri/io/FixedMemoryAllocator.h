/*!
 * @file 		FixedMemoryAllocator.h
 * @author 		Zdenek Travnicek
 * @date 		28.1.2012
 * @date		16.2.2013
 * @copyright	Institute of Intermedia, 2012 - 2013
 * 				Distributed under GNU Public License 3.0
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

#include "yuri/io/BasicIOThread.h"
#include "yuri/config/RegisteredClass.h"
#include "yuri/io/uvector.h"
namespace yuri {

namespace io {


class EXPORT FixedMemoryAllocator: public BasicIOThread {
public:
	struct Deleter {
		Deleter(yuri::size_t size, yuri::ubyte_t *original_pointer):
			size(size),original_pointer(original_pointer) {}
		Deleter(const Deleter& d):size(d.size),original_pointer(d.original_pointer) {}
		void operator()(void *mem);
		/**\brief Size of block associated with this object */
		yuri::size_t size;
		/**\brief Pointer to the memory block associated with this object */
		yuri::ubyte_t *original_pointer;

	};
	typedef std::pair<yuri::ubyte_t*, struct Deleter> memory_block_t;
	IO_THREAD_GENERATOR_DECLARATION
	static shared_ptr<Parameters> configure();
	FixedMemoryAllocator(Log &_log, pThreadBase parent, Parameters &parameters) IO_THREAD_CONSTRUCTOR;
	virtual ~FixedMemoryAllocator();
	static memory_block_t get_block(yuri::size_t size);
	static bool return_memory(yuri::size_t size, yuri::ubyte_t* mem);
	static bool allocate_blocks(yuri::size_t size, yuri::size_t count);
	static bool remove_blocks(yuri::size_t size, yuri::size_t count=0);
private:
	bool step();
	virtual bool set_param(Parameter &parameter);
	static bool do_allocate_blocks(yuri::size_t size, yuri::size_t count);

	/**\brief Global mutex protecting the pool */
	static boost::mutex mem_lock;
	/**\brief Global memory pool */
	static std::map<yuri::size_t, std::vector<yuri::ubyte_t* > > memory_pool;
	/**\brief Size of the blocks this object allocates */
	yuri::size_t block_size;
	/**\brief Number of the blocks this object allocates */
	yuri::size_t count;
};

}

}

#endif /* FIXEDMEMORYALLOCATOR_H_ */
