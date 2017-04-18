#!/usr/bin/env python
# vim:fileencoding=utf-8


from __future__ import print_function
from os.path import exists, basename, join, dirname
from os import system, rmdir, statvfs, walk
import tempfile
import warnings
from shutil import rmtree
import fnmatch


class TmpManager(object):
    '''
    A class to manage temporary files and directories.
    Use the python 'with' statement to clean-up all temporary files after use
    (at the end of the 'with' context).

    - temporary input files:
      copied locally with decompression, removed afterwards
      also decompresses tar.gz, tar.bz2 or zip archives

    - temporary output files:
      created by the program, moved to final destination upon success, removed otherwise

    - temporary files:
      created by the program, removed afterwards

    - temporary directories:
      used by the program, removed afterwards

    * Features *
        * Use a 'with' context to safely cleanup all temporary files after use,
          even when the program crashes
        * Ensure temporary files unicity by using unique temporary directories
        * Check free disk space on start
        * Choose the temporary directory

    Example:

        with TmpManager('/tmp/') as tm:  # instantiate the tmp manager on directory '/tmp/'

            # decompress a file to tmp directory and return the name
            # of the decompressed file
            input1 = tm.input('/data/file.gz')

            # if the input is an archive, returns a list of all the files in
            # the archive
            file_list = tm.input('/data/file.tar.gz')

            # returns a temporary file that will be cleaned up
            tmp = tm.file('filename.txt')

            # returns a temporary directory
            dir = tm.directory()

            # returns a filename in tmp directory
            # this file will be created afterwards, and moved to destination
            # upon commit()
            out = tm.output('/data/result.dat') 


            # move all output files to their destination
            # (otherwise they are cleared)
            tm.commit()
    '''
    def __init__(self, tmpdir='/tmp/', overwrite=False, prefix='tmpfiles_', verbose=False, freespace_mb=1000):
        self.__tmpdir = tmpdir
        self.__freespace_mb = freespace_mb
        self.__prefix = prefix
        self.__list_tmp = []   # list of temporary files
        self.__list_out = []   # list of (tmpfile, target output)
        self.__verbose = verbose
        self.__overwrite = overwrite
        assert exists(self.__tmpdir)

    def df(self, path):
        '''
        Returns the free disk space in MB
        '''
        res = statvfs(path)
        available = int(res.f_frsize*res.f_bavail/(1024.**2))   # available space in MB
        return available

    def check_free_space(self):
        '''
        Check that available space is sufficient in tmpdir
        '''
        available = self.df(self.__tmpdir)

        if (self.__freespace_mb > 0) and (available < self.__freespace_mb):
            raise IOError('Not enough free space in {} ({} MB remaining, {} MB required)'.format(
                self.__tmpdir, available, self.__freespace_mb))

    def mkdirtmp(self):
        '''
        Creates a unique directory
        '''
        tmpd = tempfile.mkdtemp(dir=self.__tmpdir, prefix=self.__prefix)
        self.__list_tmp.append(tmpd)
        return tmpd

    def remove(self, filename):
        ''' remove a file '''
        if self.__verbose:
            cmd = 'rm -fv "{}"'.format(filename)
        else:
            cmd = 'rm -f "{}"'.format(filename)
        if system(cmd):
            raise Exception('Error executing command "{}"'.format(cmd))

    def input(self, filename):
        '''
        Copy filename to temporary location
        with on-the-fly decompression, and cleanup on __exit__ (end of TmpManager context)
        '''
        self.check_free_space()
        if self.__verbose:
            v = 'v'
        else:
            v = ''

        #
        # determine how to deal with input file
        #
        if (filename.endswith('.tgz')
                or filename.endswith('.tar.gz')):
            copy = 'tar xz{v}f "{input_file}" -C "{output_dir}"'
            base = None

        elif filename.endswith('.gz'):
            copy = 'gunzip -{v}c "{input_file}" > "{output_file}"'
            base = basename(filename)[:-3]

        elif filename.endswith('.Z'):
            copy = 'gunzip -{v}c "{input_file}" > "{output_file}"'
            base = basename(filename)[:-2]

        elif filename.endswith('.tar.bz2') or filename.endswith('.tbz2'):
            copy = 'tar xj{v}f "{input_file}" -C "{output_dir}"'
            base = None

        elif filename.endswith('.tar'):
            copy = 'tar x{v}f "{input_file}" -C "{output_dir}"'
            base = None

        elif filename.endswith('.bz2'):
            copy = 'bunzip2 -{v}c "{input_file}" > "{output_file}"'
            base = basename(filename)[:-4]

        elif filename.endswith('.zip'):
            copy = 'unzip "{input_file}" -d "{output_dir}"'
            base = None

        elif cfg.verbose:
            copy = 'cp -v "{input_file}" "{output_file}"'
            base = basename(filename)
        else:
            copy = 'cp "{input_file}" "{output_file}"'
            base = basename(filename)

        # create temporary directory
        tmpd = self.mkdirtmp()

        # format the copy command
        if base is None:
            tmpfile = None
            cmd = copy.format(input_file=filename, output_dir=tmpd, v=v)
        else:
            tmpfile = join(tmpd, base)
            cmd = copy.format(input_file=filename, output_file=tmpfile, v=v)

        # does the copy
        if system(cmd):
            remove(tmpfile)
            raise IOError('Error executing "{}"'.format(cmd))

        # determine the reference file name if not done already
        if tmpfile is None:
            tmpfile = list(findfiles(tmpd, '*'))

        return tmpfile

    def output(self, target):
        '''
        Generate a temporary filename that will be moved to target location
        upon commit() (and cleaned up otherwise)
        '''
        if exists(target) and (not self.__overwrite):
            raise IOError('Error, {} exists'.format(target))

        # create output directory if necessary
        if (not exists(dirname(target))) and (dirname(target) != ''):
            cmd = 'mkdir -p {}'.format(dirname(target))
            if system(cmd):
                raise IOError('Error executing command "{}"'.format(cmd))

        self.check_free_space()

        tmpd = self.mkdirtmp()
        tmpfile = join(tmpd, basename(target))
        self.__list_out.append((tmpfile, target))

        return tmpfile

    def file(self, filename):
        '''
        Generate a temporary filename that will be cleaned up on __exit__
        '''
        self.check_free_space()

        tmpd = self.mkdirtmp()
        return join(tmpd, basename(filename))

    def directory(self):
        '''
        Generate a temporary directory
        '''
        self.check_free_space()
        return self.mkdirtmp()

    def commit(self):
        '''
        Move all output files to their target location
        '''
        while len(self.__list_out) > 0:
            (tmpfile, target) = self.__list_out.pop(0)

            if exists(target):
                if self.__overwrite:
                    self.remove(target)
                else:
                    raise IOError('Error, file {} exists'.format(target))
            if not exists(tmpfile):
                raise IOError('Error, file {} does not exist'.format(tmpfile))
            if self.__verbose:
                print('Move "{}" to "{}"'.format(tmpfile, target))
            cmd = 'mv -v "{}" "{}.tmp"'.format(tmpfile, target)
            if self.__verbose:
                print(cmd)
            if system(cmd):
                raise IOError('Error executing command "{}"'.format(cmd))
            cmd = 'mv -v "{}.tmp" "{}"'.format(target, target)
            if self.__verbose:
                print(cmd)
            if system(cmd):
                raise IOError('Error executing command "{}"'.format(cmd))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        # clean up tmpfiles
        while len(self.__list_tmp) > 0:
            tmpd = self.__list_tmp.pop(0)
            if exists(tmpd):
                rmtree(tmpd)
            else:
                warnings.warn('TmpManager.__exit__: Directory {} has already been deleted'.format(tmpd))

    def __del__(self):
        for (f, _) in self.__list_out:
            warnings.warn('"{}" has not been commited'.format(f))
        for f in self.__list_tmp:
            warnings.warn('"{}" may remain'.format(f))


# module-wide parameters
class Cfg:
    ''' store module-wise parameters
    to be used as: from tmpfiles import cfg
                   cfg.tmpdir = <some temporary directory>
                   cfg.verbose = True
                   etc.
    '''
    def __init__(self):
        # default values
        self.tmpdir = '/tmp/'  # location of the temporary directory
        self.verbose = False   # verbosity
        self.freespace = 1000  # free disk space required, in MB

        # global parameter
        self.TMPLIST = []    # a list of all 'dirty' tmpfiles

    def check_free_space(self):

        assert exists(self.tmpdir)

        if (self.freespace > 0) and (df(self.tmpdir) < self.freespace):
            raise IOError('Not enough free space in {} ({} MB remaining, {} MB required)'.format(
                self.tmpdir, df(self.tmpdir), self.freespace))
cfg = Cfg()

def df(path):
    '''
    get free disk space in MB
    '''
    res = statvfs(path)
    available = int(res.f_frsize*res.f_bavail/(1024.**2))
    return available

def findfiles(path, pat='*', split=False):
    '''
    recursively finds files starting from path and using a pattern
    if split, returns (root, filename), otherwise the full path
    '''
    if isinstance(path, list):
        paths = path
    else:
        paths = [path]
    for path in paths:
        for root, dirnames, filenames in walk(path):
            dirnames.sort()
            filenames.sort()
            for filename in fnmatch.filter(filenames, pat):
                if split:
                    yield (root, filename)
                else:
                    fname = join(root, filename)
                    yield fname


def remove(filename):
    ''' remove a file '''
    if cfg.verbose:
        system('rm -fv "{}"'.format(filename))
    else:
        system('rm -f "{}"'.format(filename))


class Tmp(str):
    '''
    A simple temporary file created by the user, removed afterwards

    Parameters:
        * tmpfile: The temporary file name (default: 'tmpfile')
          Should not contain a directory, the directory is provided module-wise
    '''
    def __new__(cls, tmpfile='tmpfile'):

        warnings.warn('The Tmp class will be deprecated. Please use TmpManager instead.')

        assert dirname(tmpfile) == ''

        # check free disk space
        cfg.check_free_space()

        tmpd = tempfile.mkdtemp(dir=cfg.tmpdir, prefix='tmpfiles_')
        tmpfile = join(tmpd, tmpfile)

        self = str.__new__(cls, tmpfile)
        self.__clean = False
        self.__tmpdir = tmpd
        self.__verbose = cfg.verbose
        cfg.TMPLIST.append(self)
        return self

    def clean(self):

        if self.__clean:
            warnings.warn('Warning, {} has already been cleaned'.format(self))
            return

        if exists(self):
            # remove temporary file
            remove(self)

        if self.__tmpdir != None:
            rmtree(self.__tmpdir)
        self.__clean = True
        cfg.TMPLIST.remove(self)


    def __del__(self):
        # raise an exception if the object is deleted before clean is called
        if not self.__clean:
            print('Warning: clean has not been called for file "{}"'.format(self))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.__clean:
            self.clean()

    @staticmethod
    def cleanAll():
        while len(cfg.TMPLIST) != 0:
            cfg.TMPLIST[0].clean()


class TmpInput(str):
    '''
    A class for simplifying the usage of temporary input files.

    Parameters
       * filename: the original file which will be copied locally and
         uncompressed if necessary
       * pattern: if the TmpInput contains multiple files, this pattern
         determines which file to use as the reference file name. If there are
         multiple files, take the first one in alphabetical order.
         default value: '*'
       * tmpdir: the base temporary directory
       * verbose

    Example:
        with TmpInput('/path/to/source.data') as f:
            # the source file is copied to a unique temporary directory
            # and f contains the file name of the temporary file
            # <use f as an input to a processor>
        
    Alternate usage:
        f = TmpInput('/path/to/source.data')
        # use f
        f.clean() # or Tmp.cleanAll()

    Note: tar files are also supported.
    '''

    # NOTE: subclassing an immutable object requires to use the __new__ method

    def __new__(cls,
            filename,
            pattern = '*'):

        warnings.warn('The TmpInput class will be deprecated. Please use TmpManager instead.')

        if cfg.verbose:
            v = 'v'
        else:
            v = ''

        # check free disk space
        cfg.check_free_space()

        #
        # determine how to deal with input file
        #
        if (filename.endswith('.tgz')
                or filename.endswith('.tar.gz')):
            copy = 'tar xz{v}f "{input_file}" -C "{output_dir}"'
            base = None

        elif filename.endswith('.gz'):
            copy = 'gunzip -{v}c "{input_file}" > "{output_file}"'
            base = basename(filename)[:-3]

        elif filename.endswith('.Z'):
            copy = 'gunzip -{v}c "{input_file}" > "{output_file}"'
            base = basename(filename)[:-2]

        elif filename.endswith('.tar.bz2') or filename.endswith('.tbz2'):
            copy = 'tar xj{v}f "{input_file}" -C "{output_dir}"'
            base = None

        elif filename.endswith('.tar'):
            copy = 'tar x{v}f "{input_file}" -C "{output_dir}"'
            base = None

        elif filename.endswith('.bz2'):
            copy = 'bunzip2 -{v}c "{input_file}" > "{output_file}"'
            base = basename(filename)[:-3]

        elif filename.endswith('.zip'):
            copy = 'unzip "{input_file}" -d "{output_dir}"'
            base = None

        elif cfg.verbose:
            copy = 'cp -v "{input_file}" "{output_file}"'
            base = basename(filename)
        else:
            copy = 'cp "{input_file}" "{output_file}"'
            base = basename(filename)

        # check that input file exists
        if not exists(filename):
            raise IOError('File "{}" does not exist'.format(filename))

        # determine temporary file name
        tmpd = tempfile.mkdtemp(dir=cfg.tmpdir, prefix='tmpfiles_')

        # format the copy command
        if base is None:
            tmpfile = None
            cmd = copy.format(input_file=filename, output_dir=tmpd, v=v)
        else:
            tmpfile = join(tmpd, base)
            cmd = copy.format(input_file=filename, output_file=tmpfile, v=v)

        # does the copy
        if system(cmd):
            remove(tmpfile)
            raise IOError('Error executing "{}"'.format(cmd))

        # determine the reference file name if not done already
        if tmpfile is None:
            tmpfile = list(findfiles(tmpd, pattern))[0]

        # create the object and sets its attributes
        self = str.__new__(cls, tmpfile)
        self.__tmpfile = tmpfile
        self.__tmpdir = tmpd
        self.__filename = filename
        self.__clean = False

        cfg.TMPLIST.append(self)

        return self

    def clean(self):

        if self.__clean:
            warnings.warn('Warning, {} has already been cleaned'.format(self))
            return

        # remove temporary directory
        rmtree(self.__tmpdir)
        self.__clean = True
        cfg.TMPLIST.remove(self)

    def source(self):
        return self.__filename

    def __del__(self):
        # raise an exception if the object is deleted before clean is called
        if not self.__clean:
            print('Warning: clean has not been called for file "{}" - temporary file "{}" may remain.'.format(self.__filename, self))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.__clean:
            self.clean()

    @staticmethod
    def cleanAll():
        while len(cfg.TMPLIST) != 0:
            cfg.TMPLIST[0].clean()

class TmpOutput(str):
    '''
    A class intended to simplify the usage of temporary output files

    Parameters:
       * filename: the target file which will be first written locally, then
         copied to the destination
       * overwrite: whether the target should be overwritten

    Example:
        with TmpOutput('/path/to/target.data') as f:
            # create f
            if <f created successfully>:
                f.move() # move f to target
        
    Alternate method:
        f = TmpOutput('/path/to/target.data')
        # at this point, f is the temporary file (non-existing yet)
        # and f.target() is the target file
        #
        # <create file f>
        #
        if <f created successfully>:
            f.move() # move f to target
        else:
            f.clean()
    '''

    # NOTE: subclassing an immutable object requires to use the __new__ method

    def __new__(cls,
            filename,
            overwrite=False):

        warnings.warn('The TmpOutput class will be deprecated. Please use TmpManager instead.')

        # check free disk space
        cfg.check_free_space()

        if cfg.verbose:
            copy = 'cp -v "{}" "{}"'
        else:
            copy = 'cp "{}" "{}"'

        # check that output file does not exist
        if not overwrite and exists(filename):
            raise IOError('File "{}" exists'.format(filename))

        # create output directory if necessary
        if (not exists(dirname(filename))) and (dirname(filename) != ''):
            system('mkdir -p {}'.format(dirname(filename)))

        # determine temporary file name
        base = basename(filename)
        tmpd = tempfile.mkdtemp(dir=cfg.tmpdir, prefix='tmpfiles_')
        tmpfile = join(tmpd, base)

        assert not exists(tmpfile)

        # create the object and sets its attributes
        self = str.__new__(cls, tmpfile)
        self.__tmpfile = tmpfile
        self.__tmpdir = tmpd
        self.__filename = filename
        self.__cp = copy
        self.__clean = False

        cfg.TMPLIST.append(self)

        return self

    def clean(self):

        if self.__clean:
            warnings.warn('Warning, {} has already been cleaned'.format(self))
            return

        # remove whole temporary directory
        rmtree(self.__tmpdir)
        self.__clean = True
        cfg.TMPLIST.remove(self)

    def target(self):
        return self.__filename

    def move(self):
        print('move', self, 'to', self.target())

        if not exists(self.__tmpfile):
            raise IOError('file {} does not exist'.format(self.__tmpfile))

        # output file: copy to (temporary) destination
        # in target folder, with extension '.tmp'
        tmpmovefile = self.__filename + '.tmp'
        cmd = self.__cp.format(self.__tmpfile, tmpmovefile)
        if system(cmd):
            raise IOError('Error executing "{}"'.format(cmd))

        # output file: rename to final file
        cmd = 'mv "{}" "{}"'.format(tmpmovefile, self.__filename)
        if system(cmd):
            raise IOError('Error executing "{}"'.format(cmd))

        self.clean()

    def __del__(self):
        # raise an exception if the object is deleted before clean is called
        if not self.__clean:
            print('Warning: clean has not been called for file "{}" - temporary file "{}" may remain.'.format(self.__filename, self))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.__clean:
            self.clean()

    @staticmethod
    def cleanAll():
        while len(cfg.TMPLIST) != 0:
            cfg.TMPLIST[0].clean()

class TmpDir(str):
    '''
    Create a temporary directory, and clean it up afterwards.

    Example:
        with TmpDir() as d:   # create a temporary directory such as /tmp/tmpfiles_9aamr0/
            # <use this directory as string d>

    Alternate:
        d = TmpDir()   # create a temporary directory such as /tmp/tmpfiles_9aamr0/
        # <use this directory as string d>
        d.clean() # remove the temporary directory
    '''

    def __new__(cls):

        warnings.warn('The TmpDir class will be deprecated. Please use TmpManager instead.')

        # check free disk space
        cfg.check_free_space()

        # create the temporary directory
        tmpd = tempfile.mkdtemp(dir=cfg.tmpdir, prefix='tmpdir_')
        ret = system('mkdir -p {}'.format(tmpd))
        if ret:
            raise 'Error creating directory {}'.format(tmpd)

        # create the object and sets its attributes
        self = str.__new__(cls, tmpd)
        self.__clean = False

        cfg.TMPLIST.append(self)

        if cfg.verbose:
            print('Creating temporary directory "{}"'.format(self))

        return self

    def clean(self):

        if cfg.verbose:
            print('Clean temporary directory "{}"'.format(self))
        rmtree(self)
        self.__clean = True
        cfg.TMPLIST.remove(self)

    def __del__(self):
        # raise an exception if the object is deleted before clean is called
        if not self.__clean:
            print('Warning: clean has not been called for file "{}"'.format(self))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if not self.__clean:
            self.clean()

    @staticmethod
    def cleanAll():
        while len(cfg.TMPLIST) != 0:
            cfg.TMPLIST[0].clean()


#
# tests
#
def test_tmp():
    cfg.verbose=True
    cfg.freespace = 1
    f = Tmp('myfile.tmp')
    open(f, 'w').write('test')
    f.clean()

def test_tmp2():
    cfg.verbose=True
    cfg.freespace = 1
    with Tmp('myfile.tmp') as f:
        open(f, 'w').write('test')

def test_input():
    cfg.verbose=True
    cfg.freespace = 1

    # create temporary file
    tmp = Tmp('myfile.tmp')
    open(tmp, 'w').write('test')
    #... and use it as a temporary input
    TmpInput(tmp)
    # clean all
    Tmp.cleanAll()

def test_output():
    cfg.verbose=True
    cfg.freespace = 10

    f = Tmp() # the target is also a temporary file
    tmp = TmpOutput(f)

    open(tmp, 'w').write('test')
    tmp.move()
    Tmp.cleanAll()

def test_dir():
    cfg.verbose=True
    cfg.freespace = 10

    d = TmpDir()
    filename = join(d, 'test')
    open(filename, 'w').write('test')
    d.clean()

if __name__ == '__main__':
    test_tmp()
    test_tmp2()
    test_input()
    test_output()
    test_dir()
