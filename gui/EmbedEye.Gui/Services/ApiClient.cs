using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using EmbedEye.Gui.Models;

namespace EmbedEye.Gui.Services;

public sealed class ApiClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;

    public ApiClient(int port, string token)
    {
        _baseUrl = $"http://127.0.0.1:{port}";
        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl),
            Timeout = TimeSpan.FromSeconds(60),
        };
        _httpClient.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", token);
    }

    public async Task<HealthResponse> GetHealthAsync()
    {
        var response = await _httpClient.GetAsync("/health");
        response.EnsureSuccessStatusCode();
        string json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<HealthResponse>(json) ?? new HealthResponse();
    }

    public async Task<SystemStatusResponse> GetStatusAsync()
    {
        var response = await _httpClient.GetAsync("/status");
        response.EnsureSuccessStatusCode();
        string json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<SystemStatusResponse>(json) ?? new SystemStatusResponse();
    }

    public async Task<FolderSummaryResponse> GetFolderSummaryAsync(string folderPath)
    {
        string encoded = Uri.EscapeDataString(folderPath);
        var response = await _httpClient.GetAsync($"/folder/summary?path={encoded}");
        response.EnsureSuccessStatusCode();
        string json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<FolderSummaryResponse>(json) ?? new FolderSummaryResponse();
    }

    public async Task<string> TriggerOcrAsync(string folderPath)
    {
        return await PostJobAsync("/jobs/ocr", folderPath);
    }

    public async Task<string> TriggerOcrCaptionAsync(string folderPath)
    {
        return await PostJobAsync("/jobs/ocr-caption", folderPath);
    }

    public async Task<string> TriggerStoreAsync(string folderPath)
    {
        return await PostJobAsync("/jobs/store", folderPath);
    }

    public async Task<string> TriggerDeleteAsync(string folderPath)
    {
        return await PostJobAsync("/jobs/delete", folderPath);
    }

    private async Task<string> PostJobAsync(string endpoint, string folderPath)
    {
        var payload = JsonSerializer.Serialize(new { folder = folderPath });
        var content = new StringContent(payload, Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync(endpoint, content);

        if ((int)response.StatusCode == 409)
        {
            throw new InvalidOperationException("Another job is currently running.");
        }

        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }

    public async Task<bool> CancelJobAsync()
    {
        var response = await _httpClient.PostAsync("/jobs/cancel", null);
        return response.IsSuccessStatusCode;
    }

    public async Task<SearchResponse> SearchAsync(string query, string mode = "visual", int topK = 3)
    {
        var payload = JsonSerializer.Serialize(new { query = query, mode = mode, top_k = topK });
        var content = new StringContent(payload, Encoding.UTF8, "application/json");
        var response = await _httpClient.PostAsync("/search", content);
        response.EnsureSuccessStatusCode();
        string json = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<SearchResponse>(json) ?? new SearchResponse();
    }

    public async Task<byte[]?> GetThumbnailBytesAsync(string path, int page = 1)
    {
        try
        {
            string encoded = Uri.EscapeDataString(path);
            var response = await _httpClient.GetAsync($"/thumbnail?path={encoded}&page={page}");
            if (!response.IsSuccessStatusCode) return null;
            return await response.Content.ReadAsByteArrayAsync();
        }
        catch
        {
            return null;
        }
    }

    public async Task<bool> RevealFileAsync(string path)
    {
        try
        {
            var payload = JsonSerializer.Serialize(new { path = path });
            var content = new StringContent(payload, Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("/reveal", content);
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public async Task ShutdownAsync()
    {
        try
        {
            await _httpClient.PostAsync("/shutdown", null);
        }
        catch
        {
            // Ignore socket disconnects during shutdown
        }
    }

    public void Dispose()
    {
        _httpClient.Dispose();
    }
}
