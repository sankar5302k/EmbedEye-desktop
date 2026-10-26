using System;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace EmbedEye.Gui.Services;

public class StageEventArgs : EventArgs
{
    public string Name { get; set; } = string.Empty;
    public int Index { get; set; }
    public int Total { get; set; }
}

public class ProgressEventArgs : EventArgs
{
    public int Done { get; set; }
    public int Total { get; set; }
    public string CurrentFile { get; set; } = string.Empty;
    public double FilesPerSec { get; set; }
    public double EtaSeconds { get; set; }
}

public class VramEventArgs : EventArgs
{
    public double AllocatedMb { get; set; }
    public double ReservedMb { get; set; }
    public double TotalMb { get; set; }
    public string[] ResidentModels { get; set; } = Array.Empty<string>();
}

public sealed class WebSocketClient : IDisposable
{
    private ClientWebSocket? _webSocket;
    private readonly Uri _serverUri;
    private readonly CancellationTokenSource _cts = new();
    private bool _isDisposed;

    public bool IsConnected => _webSocket?.State == WebSocketState.Open;

    public event Action<bool>? ConnectionStateChanged;
    public event Action<string, string>? LogReceived;
    public event Action<StageEventArgs>? StageChanged;
    public event Action<ProgressEventArgs>? ProgressUpdated;
    public event Action<VramEventArgs>? VramUpdated;
    public event Action<string>? JobStarted;
    public event Action<string>? JobFinished;
    public event Action<string>? JobFailed;
    public event Action? JobCancelled;

    public WebSocketClient(int port, string token)
    {
        _serverUri = new Uri($"ws://127.0.0.1:{port}/ws/events?token={Uri.EscapeDataString(token)}");
    }

    public void Start()
    {
        Task.Run(RunReceiveLoopAsync);
    }

    private async Task RunReceiveLoopAsync()
    {
        byte[] buffer = new byte[65536];
        int retryDelayMs = 500;

        while (!_cts.IsCancellationRequested)
        {
            try
            {
                _webSocket?.Dispose();
                _webSocket = new ClientWebSocket();
                await _webSocket.ConnectAsync(_serverUri, _cts.Token);
                ConnectionStateChanged?.Invoke(true);
                retryDelayMs = 500; // Reset delay on successful connection

                while (_webSocket.State == WebSocketState.Open && !_cts.IsCancellationRequested)
                {
                    var result = await _webSocket.ReceiveAsync(new ArraySegment<byte>(buffer), _cts.Token);
                    if (result.MessageType == WebSocketMessageType.Close)
                    {
                        break;
                    }

                    string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                    HandleEventMessage(message);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                LoggerService.Warn($"WebSocket error: {ex.Message}");
            }

            ConnectionStateChanged?.Invoke(false);

            if (!_cts.IsCancellationRequested)
            {
                await Task.Delay(retryDelayMs);
                retryDelayMs = Math.Min(retryDelayMs * 2, 5000);
            }
        }
    }

    private void HandleEventMessage(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (!root.TryGetProperty("event", out var evProp)) return;
            string eventType = evProp.GetString() ?? "";

            root.TryGetProperty("data", out var data);

            switch (eventType)
            {
                case "job_started":
                    string jt = data.TryGetProperty("job_type", out var jp) ? jp.GetString() ?? "" : "";
                    JobStarted?.Invoke(jt);
                    break;

                case "stage_changed":
                    StageChanged?.Invoke(new StageEventArgs
                    {
                        Name = data.TryGetProperty("name", out var np) ? np.GetString() ?? "" : "",
                        Index = data.TryGetProperty("index", out var ip) ? ip.GetInt32() : 1,
                        Total = data.TryGetProperty("total", out var tp) ? tp.GetInt32() : 1,
                    });
                    break;

                case "progress":
                    ProgressUpdated?.Invoke(new ProgressEventArgs
                    {
                        Done = data.TryGetProperty("done", out var dp) ? dp.GetInt32() : 0,
                        Total = data.TryGetProperty("total", out var totp) ? totp.GetInt32() : 0,
                        CurrentFile = data.TryGetProperty("current_file", out var cfp) ? cfp.GetString() ?? "" : "",
                        FilesPerSec = data.TryGetProperty("files_per_sec", out var fpsp) ? fpsp.GetDouble() : 0,
                        EtaSeconds = data.TryGetProperty("eta_seconds", out var etap) ? etap.GetDouble() : 0,
                    });
                    break;

                case "log":
                    string lvl = data.TryGetProperty("level", out var lp) ? lp.GetString() ?? "INFO" : "INFO";
                    string msg = data.TryGetProperty("message", out var mp) ? mp.GetString() ?? "" : "";
                    LogReceived?.Invoke(lvl, msg);
                    break;

                case "vram":
                    var residentList = new System.Collections.Generic.List<string>();
                    if (data.TryGetProperty("resident", out var resArr) && resArr.ValueKind == JsonValueKind.Array)
                    {
                        foreach (var el in resArr.EnumerateArray())
                        {
                            string? item = el.GetString();
                            if (!string.IsNullOrEmpty(item)) residentList.Add(item);
                        }
                    }
                    VramUpdated?.Invoke(new VramEventArgs
                    {
                        AllocatedMb = data.TryGetProperty("allocated", out var ap) ? ap.GetDouble() : 0,
                        ReservedMb = data.TryGetProperty("reserved", out var rp) ? rp.GetDouble() : 0,
                        TotalMb = data.TryGetProperty("total", out var top) ? top.GetDouble() : 0,
                        ResidentModels = residentList.ToArray(),
                    });
                    break;

                case "job_finished":
                    JobFinished?.Invoke(data.ToString());
                    break;

                case "job_failed":
                    string failMsg = data.TryGetProperty("message", out var fmp) ? fmp.GetString() ?? "Job failed" : "Job failed";
                    JobFailed?.Invoke(failMsg);
                    break;

                case "job_cancelled":
                    JobCancelled?.Invoke();
                    break;
            }
        }
        catch (Exception ex)
        {
            LoggerService.Warn($"Failed parsing event message: {ex.Message}");
        }
    }

    public void Dispose()
    {
        if (!_isDisposed)
        {
            _cts.Cancel();
            _webSocket?.Dispose();
            _cts.Dispose();
            _isDisposed = true;
        }
    }
}
