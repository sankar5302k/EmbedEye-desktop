using System;
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace EmbedEye.Gui.Services;

/// <summary>
/// Manages a Windows Job Object configured with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE.
/// Any child process assigned to this Job Object will be automatically terminated
/// by the Windows OS kernel when this parent process closes or terminates unexpectedly.
/// </summary>
public sealed class JobObjectService : IDisposable
{
    private IntPtr _jobHandle = IntPtr.Zero;
    private bool _disposed;

    public JobObjectService()
    {
        if (OperatingSystem.IsWindows())
        {
            InitializeJobObject();
        }
    }

    private void InitializeJobObject()
    {
        _jobHandle = CreateJobObject(IntPtr.Zero, null);
        if (_jobHandle == IntPtr.Zero)
        {
            return;
        }

        var info = new JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        };

        var extendedInfo = new JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            BasicLimitInformation = info
        };

        int length = Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION));
        IntPtr extendedInfoPtr = Marshal.AllocHGlobal(length);
        try
        {
            Marshal.StructureToPtr(extendedInfo, extendedInfoPtr, false);
            SetInformationJobObject(
                _jobHandle,
                JobObjectInfoType.ExtendedLimitInformation,
                extendedInfoPtr,
                (uint)length);
        }
        finally
        {
            Marshal.FreeHGlobal(extendedInfoPtr);
        }
    }

    /// <summary>
    /// Assigns a running process to this Job Object.
    /// </summary>
    public bool AssignProcess(Process process)
    {
        if (!OperatingSystem.IsWindows() || _jobHandle == IntPtr.Zero || process == null)
            return false;

        try
        {
            return AssignProcessToJobObject(_jobHandle, process.Handle);
        }
        catch
        {
            return false;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_jobHandle != IntPtr.Zero)
            {
                CloseHandle(_jobHandle);
                _jobHandle = IntPtr.Zero;
            }
            _disposed = true;
        }
    }

    #region Win32 Native Interop
    private const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;

    private enum JobObjectInfoType
    {
        ExtendedLimitInformation = 9
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct IO_COUNTERS
    {
        public ulong ReadOperationCount;
        public ulong WriteOperationCount;
        public ulong OtherOperationCount;
        public ulong ReadTransferCount;
        public ulong WriteTransferCount;
        public ulong OtherTransferCount;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
    {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
    {
        public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
        public IO_COUNTERS IoInfo;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryLimit;
        public UIntPtr PeakJobMemoryLimit;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr lpJobAttributes, string? lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(
        IntPtr hJob,
        JobObjectInfoType JobObjectInformationClass,
        IntPtr lpJobObjectInformation,
        uint cbJobObjectInformationLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr hJob, IntPtr hProcess);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);
    #endregion
}
