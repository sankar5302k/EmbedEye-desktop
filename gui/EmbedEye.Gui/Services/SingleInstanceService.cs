using System;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace EmbedEye.Gui.Services;

public sealed class SingleInstanceService : IDisposable
{
    private const string MutexName = "EmbedEye_SingleInstance_Mutex_d2ba65";
    private const string PipeName = "EmbedEye_SingleInstance_Pipe_d2ba65";

    private Mutex? _mutex;
    private bool _hasHandle;
    private readonly CancellationTokenSource _cts = new();

    public event Action<string>? FolderArgumentReceived;

    /// <summary>
    /// Check if this instance is the primary instance.
    /// If secondary, sends folder argument to primary instance and returns false.
    /// </summary>
    public bool TryAcquire(string[] args)
    {
        _mutex = new Mutex(true, MutexName, out _hasHandle);

        if (!_hasHandle)
        {
            // Another instance is already running; forward argument
            string folderArg = args.Length > 0 ? args[0] : "";
            ForwardArgumentToPrimary(folderArg);
            return false;
        }

        // Primary instance: start named pipe listener
        StartPipeListener();
        return true;
    }

    private void StartPipeListener()
    {
        Task.Run(async () =>
        {
            while (!_cts.IsCancellationRequested)
            {
                try
                {
                    using var server = new NamedPipeServerStream(
                        PipeName,
                        PipeDirection.In,
                        1,
                        PipeTransmissionMode.Byte,
                        PipeOptions.Asynchronous);

                    await server.WaitForConnectionAsync(_cts.Token);

                    using var reader = new StreamReader(server, Encoding.UTF8);
                    string? receivedFolder = await reader.ReadLineAsync();

                    if (!string.IsNullOrEmpty(receivedFolder))
                    {
                        FolderArgumentReceived?.Invoke(receivedFolder);
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    LoggerService.Warn($"Named pipe server error: {ex.Message}");
                }
            }
        });
    }

    private static void ForwardArgumentToPrimary(string folder)
    {
        try
        {
            using var client = new NamedPipeClientStream(".", PipeName, PipeDirection.Out);
            client.Connect(1500); // 1.5s timeout
            using var writer = new StreamWriter(client, Encoding.UTF8);
            writer.WriteLine(folder);
            writer.Flush();
        }
        catch (Exception ex)
        {
            LoggerService.Warn($"Failed to forward argument to primary instance: {ex.Message}");
        }
    }

    public void Dispose()
    {
        _cts.Cancel();
        _cts.Dispose();

        if (_hasHandle && _mutex != null)
        {
            try
            {
                _mutex.ReleaseMutex();
            }
            catch
            {
            }
            _mutex.Dispose();
            _hasHandle = false;
        }
    }
}
