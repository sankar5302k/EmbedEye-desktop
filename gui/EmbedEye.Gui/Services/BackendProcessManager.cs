using System;
using System.Diagnostics;
using System.IO;
using System.Security.Cryptography;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace EmbedEye.Gui.Services;

public sealed class BackendProcessManager : IDisposable
{
    private Process? _process;
    private readonly JobObjectService _jobObjectService;
    private readonly AppSettings _settings;
    private readonly TaskCompletionSource<int> _readyPortTcs = new();
    private bool _isDisposed;
    private bool _isExpectedExit;

    public string AuthToken { get; private set; } = string.Empty;
    public int BoundPort { get; private set; } = 0;
    public bool IsRunning => _process != null && !_process.HasExited;

    public event Action<string>? Ready;
    public event Action<string>? LogReceived;
    public event Action<string>? BackendCrashed;

    public BackendProcessManager(AppSettings settings)
    {
        _settings = settings;
        _jobObjectService = new JobObjectService();
    }

    /// <summary>
    /// Launches backend python process, assigns to Job Object, and waits for ready line.
    /// </summary>
    public async Task<int> StartAsync(CancellationToken cancellationToken = default)
    {
        // Generate cryptographic random 32-byte session token (hex encoded: 64 chars)
        byte[] tokenBytes = new byte[32];
        RandomNumberGenerator.Fill(tokenBytes);
        AuthToken = Convert.ToHexString(tokenBytes).ToLowerInvariant();

        string pythonExe = _settings.PythonPath;
        string repoRoot = _settings.RepoRoot;

        if (!File.Exists(pythonExe))
        {
            throw new FileNotFoundException(
                $"Python executable not found at '{pythonExe}'. Please verify Python or your venv.",
                pythonExe);
        }

        if (!Directory.Exists(repoRoot))
        {
            throw new DirectoryNotFoundException(
                $"EmbedEye repository root not found at '{repoRoot}'.");
        }

        LoggerService.Info($"Launching backend: {pythonExe} -m backend.server in {repoRoot}");

        var startInfo = new ProcessStartInfo
        {
            FileName = pythonExe,
            Arguments = "-u -m backend.server",
            WorkingDirectory = repoRoot,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        // Pass authentication token and Python unbuffered flag
        startInfo.EnvironmentVariables["EMBEDEYE_TOKEN"] = AuthToken;
        startInfo.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
        startInfo.EnvironmentVariables["PYTHONUTF8"] = "1";

        _isExpectedExit = false;
        _process = new Process { StartInfo = startInfo, EnableRaisingEvents = true };

        _process.OutputDataReceived += (s, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                HandleStdoutLine(e.Data);
            }
        };

        _process.ErrorDataReceived += (s, e) =>
        {
            if (!string.IsNullOrEmpty(e.Data))
            {
                LoggerService.Warn($"[Backend Stderr] {e.Data}");
                LogReceived?.Invoke(e.Data);
            }
        };

        _process.Exited += (s, e) =>
        {
            int exitCode = _process?.ExitCode ?? -1;
            LoggerService.Warn($"Backend process exited with code {exitCode}");
            if (!_isExpectedExit)
            {
                BackendCrashed?.Invoke($"Engine process terminated unexpectedly (code {exitCode}).");
            }
        };

        if (!_process.Start())
        {
            throw new InvalidOperationException("Failed to start python backend process.");
        }

        // Attach to Windows Job Object immediately
        _jobObjectService.AssignProcess(_process);

        _process.BeginOutputReadLine();
        _process.BeginErrorReadLine();

        // Wait for ready event JSON
        using var registration = cancellationToken.Register(() => _readyPortTcs.TrySetCanceled());
        int port = await _readyPortTcs.Task;
        BoundPort = port;
        LoggerService.Info($"Backend successfully bound to port {port}");
        return port;
    }

    private void HandleStdoutLine(string line)
    {
        LoggerService.Info($"[Backend Stdout] {line}");
        LogReceived?.Invoke(line);

        if (!_readyPortTcs.Task.IsCompleted)
        {
            try
            {
                using var doc = JsonDocument.Parse(line);
                var root = doc.RootElement;
                if (root.TryGetProperty("event", out var ev) && ev.GetString() == "ready" &&
                    root.TryGetProperty("port", out var portProp))
                {
                    int port = portProp.GetInt32();
                    _readyPortTcs.TrySetResult(port);
                    Ready?.Invoke($"Port {port}");
                }
            }
            catch
            {
                // Non-JSON stdout lines before ready
            }
        }
    }

    /// <summary>
    /// Gracefully shutdown backend process.
    /// </summary>
    public async Task StopAsync()
    {
        _isExpectedExit = true;
        if (_process != null && !_process.HasExited)
        {
            try
            {
                _process.Kill();
                await _process.WaitForExitAsync();
            }
            catch
            {
                // Ignore errors during exit
            }
        }
    }

    public void Dispose()
    {
        if (!_isDisposed)
        {
            _isExpectedExit = true;
            try
            {
                if (_process != null && !_process.HasExited)
                {
                    _process.Kill();
                }
            }
            catch
            {
            }

            _jobObjectService.Dispose();
            _process?.Dispose();
            _isDisposed = true;
        }
    }
}
