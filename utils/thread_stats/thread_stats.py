'''
Created on Jan 27, 2011

@author: worker
'''
import os, os.path
import exceptions
import time
import Gnuplot
class ThreadStats(object):
    '''
    classdocs
    '''


    def __init__(self,pid,out_file):
        '''
        Constructor
        '''
        #self._pid = pid
        #self._dir = os.path.join('/proc',str(self._pid),'task')
        self._out_file = out_file
        self._stats = {}
        self._task_list = []
        self._task_stats = {}
        self._task_names = {}
        self._g = None
        #if not os.path.isdir(self._dir):
        self._pid = self.find_pid(pid)
        self._dir = None
        if self._pid: 
            self._dir = os.path.join('/proc',str(self._pid),'task')
        if not self._dir or not os.path.isdir(self._dir):
            raise exceptions.Exception('No such a process %s' % str(pid))
        #try:
        #    self._out = open(self._out_file,'wb')
        #except IOError as e:
        #    raise exceptions.Exception('Failed to open output file %s (%s)' % (self._out_file, str(e)))
            
        self._name = open(os.path.join('/proc',str(self._pid),'comm')).read()
        print ('Processing %s' % self._name)
        
    
    def parse_stat(self):
        tasks = os.listdir(self._dir)
        vals = {}
        if not os.path.isdir(self._dir):
            raise exceptions.Exception('No such a process %d' % self._pid)
        for task in tasks:
            task_dir = os.path.join(self._dir,task)
            if not os.path.isdir(task_dir):
                continue
            #print 'Processing %s' % task_dir
            task_stat = os.path.join(task_dir,'stat')
            try:
                task_data = open(task_stat).read().split()
            except IOError as e:
                print ('Failed to parse stats for %s (%s)' % (task,str(e)))
                continue
            #print task_data
            td = {'state':task_data[2],
                  'utime':int(task_data[13]),
                  'stime':int(task_data[14])}
            vals[task]=td
            if not task in self._task_list:
                self._task_list.append(task)          
        return vals
    def load_names(self):
        for i in self._task_list:
            if not i in self._task_names:
                try:
                    name = open('/tmp/%d/%d'%(int(self._pid),int(i)),'rt').read()
                except:
                    continue
                self._task_names[i]=name
        
        
            
    def record_stats(self,delay=1):
        self.stats = {}
        start_time = time.time()
        try:
            while True:
                t = time.time() - start_time
                try:
                    vals = self.parse_stat()
                    self.load_names()
                except exceptions.Exception:
                    break
                
                self._stats[t]=vals
                for i in vals:
                    #print self._task_stats
                    if i not in self._task_stats:
                        self._task_stats[i] = {'utime':{},'last':t,'user':[], 'stime':{},'sys':[],'total':[]}
                    else:
                        if len(self._task_stats[i]['user']) > 0:
                            lasttime =  self._task_stats[i]['user'][-10:][0][0]
                        else:
                            lasttime = self._task_stats[i]['last']
                        
                        difft = (float(vals[i]['utime']) - float(self._task_stats[i]['utime'][lasttime])) / (t - lasttime)
                        diffs = (float(vals[i]['stime']) - float(self._task_stats[i]['stime'][lasttime])) / (t - lasttime)
                        #print ('%s -> %s, %s' % (float(self._task_stats[i]['utime'][lasttime]),str(float(vals[i]['utime'])),str(difft)))
                        self._task_stats[i]['user'].append([t,difft])
                        self._task_stats[i]['sys'].append([t,diffs])
                        self._task_stats[i]['total'].append([t,difft+diffs])
                        self._task_stats[i]['last']=t
                    self._task_stats[i]['utime'][t]=vals[i]['utime']
                    self._task_stats[i]['stime'][t]=vals[i]['stime']
                    
                self.plot_stats()
                time.sleep(delay)
        except KeyboardInterrupt:
                pass
        return self._stats
    def dump_stats(self):
        header = 'Time'
        for tid in self._task_list:
            header += '\t%s User\t%s System' % (tid,tid)
        header+='\n'
        self._out.write(header)
        for t in sorted(self._stats):
            line = '%s' % str(t)
            tstat = self._stats[t]
            for tid in self._task_list:
                if tid in tstat:
                    line += '\t%s\t%s' % (tstat[tid]['utime'],tstat[tid]['stime'])
                else:
                    line += '\t\t'
            line += '\n'
            self._out.write(line)
           
    def plot_stats(self):
        if not self._g:
            self._g = Gnuplot.Gnuplot()
            self._g.title('Thread statistics')
            self._g('set terminal x11')

        plots = []
        for i in self._task_stats:
            if not 'user' in self._task_stats[i] or len(self._task_stats[i]['user']) < 2:
                continue
            #p = Gnuplot.PlotItems.Data([[1,1],[2,1],[3,5],[4,2]],with_='lines',title='Thread USER')
            #g.plot(p)
            #return False
            #p = Gnuplot.PlotItems.Data(self._task_stats[i]['user'][-100:],with_='linespoints',title='Thread %s USER' % str(i))
            #plots.append(p)            
            #p = Gnuplot.PlotItems.Data(self._task_stats[i]['sys'][-100:],with_='linespoints',title='Thread %s SYSTEM' % str(i))
            #plots.append(p)
            thid = self._task_names[i] if i in self._task_names else 'Thread %d' % int(i)
            thid = thid.strip()
            p = Gnuplot.PlotItems.Data(self._task_stats[i]['total'][-100:],with_='linespoints',title='%s TOTAL' % (thid))
            plots.append(p)
            #print self._task_stats[i]['user']


        if len(plots) == 0:
            return 
        self._g.reset()
        self._g('set xlabel  "time [s]"')
        self._g('set ylabel  "cputime [used jiffies/s]"')
        self._g.replot(*plots)
        
    def dump_plot(self):
        plots = []
        for i in self._task_stats:
            p = Gnuplot.PlotItems.Data(self._task_stats[i]['user'],with_='lines',title='Thread %s USER' % str(i))
            plots.append(p)
        g2 = Gnuplot.Gnuplot()
        g2.title('Thread statistics')
        g2('set terminal png')
        g2('set xlabel  "time [s]"')
        g2('set ylabel  "cputime [used jiffies/s]"')
        g2('set output "%s"' % self._out_file)
        g2.plot(*plots)
            
    def find_pid(self,name):
        for i in os.listdir('/proc'):
            path = os.path.join('/proc',i,'comm')
            if os.path.isfile(path):
                try:
                    if open(path).read().strip() == name:
                        return int(i)
                except:
                    continue
        return None
                
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print ('Usage: %s PID dump_file' % sys.argv[0])
        sys.exit(1)
    ts = ThreadStats(sys.argv[1],sys.argv[2])
    ts.record_stats(0.2)
    ts.dump_plot()
    
    
    
    