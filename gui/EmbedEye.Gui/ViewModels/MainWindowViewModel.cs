using System;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EmbedEye.Gui.Models;
using EmbedEye.Gui.Services;

namespace EmbedEye.Gui.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
    private readonly SettingsService _settingsService;
    private BackendProcessManager? _backendManager;
    private ApiClient? _apiClient;
    private WebSocketClient? _webSocketClient;
    private CancellationTokenSource? _folderValidationCts;

    // =========================================================================
    // SCREEN STATE
    // =========================================================================
    [ObservableProperty]
    private bool _isSplashVisible = true;

    [ObservableProperty]
    private string _splashStatusText = "Starting EmbedEye engine...";

    [ObservableProperty]
    private double _splashProgress = 15;

    [ObservableProperty]
    private bool _backendCrashed = false;

    [ObservableProperty]
    private string _backendCrashMessage = string.Empty;

    // =========================================================================
    // NAVIGATION TABS
    // =========================================================================
    [ObservableProperty]
    private int _selectedTabIndex = 0; // 0 = Search (default), 1 = Index & Manage

    public bool IsSearchTabSelected => SelectedTabIndex == 0;
    public bool IsIndexTabSelected => SelectedTabIndex == 1;

    partial void OnSelectedTabIndexChanged(int value)
    {
        OnPropertyChanged(nameof(IsSearchTabSelected));
        OnPropertyChanged(nameof(IsIndexTabSelected));
    }

    // =========================================================================
    // SEARCH FUNCTIONALITY (CORE FEATURE - TOP 3 RETRIEVAL)
    // =========================================================================
    [ObservableProperty]
    private string _searchQuery = string.Empty;

    [ObservableProperty]
    private string _searchMode = "visual"; // "visual" or "ocr"

    [ObservableProperty]
    private bool _isVisualMode = true;

    [ObservableProperty]
    private bool _isOcrMode = false;

    partial void OnIsVisualModeChanged(bool value)
    {
        if (value)
        {
            _isOcrMode = false;
            SearchMode = "visual";
            OnPropertyChanged(nameof(IsOcrMode));
        }
    }

    partial void OnIsOcrModeChanged(bool value)
    {
        if (value)
        {
            _isVisualMode = false;
            SearchMode = "ocr";
            OnPropertyChanged(nameof(IsVisualMode));
        }
    }

    [ObservableProperty]
    private bool _isSearching = false;

    [ObservableProperty]
    private bool _hasSearched = false;

    [ObservableProperty]
    private string _searchBadge = string.Empty;

    [ObservableProperty]
    private string _searchStatusText = string.Empty;

    public ObservableCollection<SearchResultItemModel> SearchResults { get; } = new();

    public bool HasSearchResults => SearchResults.Count > 0;
    public bool NoResultsFound => HasSearched && !IsSearching && SearchResults.Count == 0;

    // =========================================================================
    // FOLDER INPUT & VALIDATION
    // =========================================================================
    [ObservableProperty]
    private string _folderInput = string.Empty;

    [ObservableProperty]
    private bool _isFolderValid = false;

    [ObservableProperty]
    private string _folderValidationMessage = string.Empty;

    [ObservableProperty]
    private bool _isSummaryLoading = false;

    [ObservableProperty]
    private int _imagesFoundCount = 0;

    [ObservableProperty]
    private int _docsFoundCount = 0;

    [ObservableProperty]
    private int _stagedOcrCount = 0;

    [ObservableProperty]
    private int _stagedCaptionCount = 0;

    [ObservableProperty]
    private int _storedPhotosCount = 0;

    [ObservableProperty]
    private int _storedDocsCount = 0;

    // =========================================================================
    // ACTION BUTTON STATES & JOBS
    // =========================================================================
    [ObservableProperty]
    private bool _isJobRunning = false;

    [ObservableProperty]
    private string _activeJobType = string.Empty;

    [ObservableProperty]
    private bool _isOcrActive = false;

    [ObservableProperty]
    private bool _isOcrCaptionActive = false;

    [ObservableProperty]
    private bool _isStoreActive = false;

    [ObservableProperty]
    private bool _isDeleteActive = false;

    // =========================================================================
    // PROGRESS CARD
    // =========================================================================
    [ObservableProperty]
    private bool _isProgressVisible = false;

    [ObservableProperty]
    private string _stageTitle = "Ready";

    [ObservableProperty]
    private int _stageIndex = 1;

    [ObservableProperty]
    private int _totalStages = 1;

    [ObservableProperty]
    private double _progressPercent = 0;

    [ObservableProperty]
    private bool _isIndeterminate = false;

    [ObservableProperty]
    private string _currentFileText = string.Empty;

    [ObservableProperty]
    private string _rateText = "0 files/s";

    [ObservableProperty]
    private string _etaText = "ETA: --";

    // =========================================================================
    // VRAM GAUGE & RESIDENT MODELS
    // =========================================================================
    [ObservableProperty]
    private string _deviceName = "Detecting device...";

    [ObservableProperty]
    private double _vramAllocatedMb = 0;

    [ObservableProperty]
    private double _vramTotalMb = 4096;

    [ObservableProperty]
    private double _vramPercent = 0;

    [ObservableProperty]
    private string _vramText = "0 / 0 MB";

    public ObservableCollection<string> ResidentModels { get; } = new();

    // =========================================================================
    // LOGS
    // =========================================================================
    public ObservableCollection<LogEntryModel> Logs { get; } = new();

    // =========================================================================
    // RESULT BANNER / TOAST
    // =========================================================================
    [ObservableProperty]
    private string _resultStatus = "None"; // None, Success, Failed, Cancelled

    [ObservableProperty]
    private string _resultMessage = string.Empty;

    // =========================================================================
    // DELETE MODAL
    // =========================================================================
    [ObservableProperty]
    private bool _isDeleteModalOpen = false;

    [ObservableProperty]
    private string _deleteModalText = string.Empty;

    [ObservableProperty]
    private double _deleteHoldProgress = 0;

    // =========================================================================
    // SETTINGS FLYOUT
    // =========================================================================
    [ObservableProperty]
    private bool _reduceAnimations = false;

    [ObservableProperty]
    private bool _isContextMenuRegistered = false;

    public MainWindowViewModel()
    {
        _settingsService = new SettingsService();
        _reduceAnimations = _settingsService.Current.ReduceAnimations;
        _isContextMenuRegistered = ContextMenuService.IsRegistered();
    }

    public async Task InitializeAsync(string? initialFolder = null)
    {
        if (!string.IsNullOrWhiteSpace(initialFolder))
        {
            FolderInput = initialFolder;
        }

        await StartBackendAndConnectAsync();
    }

    public async Task StartBackendAndConnectAsync()
    {
        BackendCrashed = false;
        IsSplashVisible = true;
        SplashStatusText = "Starting EmbedEye backend engine...";
        SplashProgress = 20;

        try
        {
            _backendManager?.Dispose();
            _backendManager = new BackendProcessManager(_settingsService.Current);
            _backendManager.BackendCrashed += OnBackendCrashed;

            int port = await _backendManager.StartAsync();

            SplashStatusText = "Connecting and loading libraries...";
            SplashProgress = 50;

            _apiClient?.Dispose();
            _apiClient = new ApiClient(port, _backendManager.AuthToken);

            // Poll /health until ready
            for (int i = 0; i < 30; i++)
            {
                try
                {
                    var health = await _apiClient.GetHealthAsync();
                    SplashStatusText = string.IsNullOrWhiteSpace(health.Detail) ? "Initializing..." : health.Detail;
                    SplashProgress = Math.Min(90, 50 + (i * 3));

                    if (health.State == "ready")
                        break;
                }
                catch
                {
                    // Backend still spinning up
                }
                await Task.Delay(200);
            }

            SplashStatusText = "Ready";
            SplashProgress = 100;
            await Task.Delay(300);

            // Connect WebSocket
            _webSocketClient?.Dispose();
            _webSocketClient = new WebSocketClient(port, _backendManager.AuthToken);
            HookWebSocketEvents();
            _webSocketClient.Start();

            // Fetch initial status
            await RefreshStatusAsync();

            // Transition to Main Window
            IsSplashVisible = false;

            if (!string.IsNullOrWhiteSpace(FolderInput))
            {
                await ValidateAndSummarizeFolderAsync(FolderInput);
            }
        }
        catch (Exception ex)
        {
            LoggerService.Error("Failed to initialize backend", ex);
            OnBackendCrashed($"Startup error: {ex.Message}");
        }
    }

    private void OnBackendCrashed(string reason)
    {
        Dispatcher.UIThread.Post(() =>
        {
            BackendCrashed = true;
            BackendCrashMessage = reason;
            IsJobRunning = false;
            ResetActiveActionStates();
        });
    }

    private void HookWebSocketEvents()
    {
        if (_webSocketClient == null) return;

        _webSocketClient.LogReceived += (level, message) =>
        {
            Dispatcher.UIThread.Post(() =>
            {
                Logs.Add(new LogEntryModel { Level = level, Message = message });
                if (Logs.Count > 500) Logs.RemoveAt(0);
            });
        };

        _webSocketClient.StageChanged += args =>
        {
            Dispatcher.UIThread.Post(() =>
            {
                StageTitle = args.Name;
                StageIndex = args.Index;
                TotalStages = args.Total;
            });
        };

        _webSocketClient.ProgressUpdated += args =>
        {
            Dispatcher.UIThread.Post(() =>
            {
                if (args.Total > 0)
                {
                    IsIndeterminate = false;
                    ProgressPercent = Math.Clamp(((double)args.Done / args.Total) * 100.0, 0, 100);
                }
                else
                {
                    IsIndeterminate = true;
                }

                CurrentFileText = args.CurrentFile;
                RateText = $"{args.FilesPerSec:F1} files/s";
                EtaText = args.EtaSeconds > 0 ? $"ETA: {args.EtaSeconds:F0}s" : "ETA: --";
            });
        };

        _webSocketClient.VramUpdated += args =>
        {
            Dispatcher.UIThread.Post(() =>
            {
                VramAllocatedMb = args.AllocatedMb;
                VramTotalMb = args.TotalMb > 0 ? args.TotalMb : 4096;
                VramPercent = Math.Clamp((args.AllocatedMb / VramTotalMb) * 100.0, 0, 100);
                VramText = $"{args.AllocatedMb:F0} / {VramTotalMb:F0} MB";

                ResidentModels.Clear();
                foreach (var m in args.ResidentModels)
                {
                    ResidentModels.Add(m);
                }
            });
        };

        _webSocketClient.JobFinished += summaryJson =>
        {
            Dispatcher.UIThread.Post(async () =>
            {
                IsJobRunning = false;
                IsIndeterminate = false;
                ProgressPercent = 100;
                CurrentFileText = "Operation completed";
                RateText = string.Empty;
                EtaText = string.Empty;
                ResetActiveActionStates();
                StageTitle = "Completed";
                ResultStatus = "Success";
                ResultMessage = "Operation finished successfully.";

                await RefreshStatusAsync();
                if (IsFolderValid)
                {
                    await ValidateAndSummarizeFolderAsync(FolderInput);
                }

                _ = Task.Run(async () =>
                {
                    await Task.Delay(2500);
                    await Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        if (!IsJobRunning)
                        {
                            IsProgressVisible = false;
                        }
                    });
                });
            });
        };

        _webSocketClient.JobFailed += message =>
        {
            Dispatcher.UIThread.Post(async () =>
            {
                IsJobRunning = false;
                IsIndeterminate = false;
                ResetActiveActionStates();
                ResultStatus = "Failed";
                ResultMessage = message;
                await RefreshStatusAsync();
            });
        };

        _webSocketClient.JobCancelled += () =>
        {
            Dispatcher.UIThread.Post(async () =>
            {
                IsJobRunning = false;
                IsIndeterminate = false;
                ResetActiveActionStates();
                ResultStatus = "Cancelled";
                ResultMessage = "Operation was cancelled by user.";
                await RefreshStatusAsync();
            });
        };
    }

    private void ResetActiveActionStates()
    {
        IsOcrActive = false;
        IsOcrCaptionActive = false;
        IsStoreActive = false;
        IsDeleteActive = false;
        ActiveJobType = string.Empty;
    }

    // =========================================================================
    // FOLDER INPUT HANDLERS
    // =========================================================================
    partial void OnFolderInputChanged(string value)
    {
        _folderValidationCts?.Cancel();
        _folderValidationCts = new CancellationTokenSource();
        var token = _folderValidationCts.Token;

        Task.Run(async () =>
        {
            await Task.Delay(250, token); // Debounce
            if (!token.IsCancellationRequested)
            {
                await ValidateAndSummarizeFolderAsync(value);
            }
        }, token);
    }

    public async Task ValidateAndSummarizeFolderAsync(string rawPath)
    {
        if (string.IsNullOrWhiteSpace(rawPath))
        {
            Dispatcher.UIThread.Post(() =>
            {
                IsFolderValid = false;
                FolderValidationMessage = string.Empty;
                ClearSummaryCounts();
            });
            return;
        }

        string trimmed = rawPath.Trim();
        if ((trimmed.StartsWith("\"") && trimmed.EndsWith("\"")) || (trimmed.StartsWith("'") && trimmed.EndsWith("'")))
        {
            trimmed = trimmed.Substring(1, trimmed.Length - 2).Trim();
        }

        string expanded = Environment.ExpandEnvironmentVariables(trimmed);

        if (!Directory.Exists(expanded))
        {
            Dispatcher.UIThread.Post(() =>
            {
                IsFolderValid = false;
                FolderValidationMessage = "Folder does not exist";
                ClearSummaryCounts();
            });
            return;
        }

        Dispatcher.UIThread.Post(() =>
        {
            IsFolderValid = true;
            FolderValidationMessage = "Valid folder";
            IsSummaryLoading = true;
        });

        try
        {
            if (_apiClient != null)
            {
                var summary = await _apiClient.GetFolderSummaryAsync(expanded);
                Dispatcher.UIThread.Post(() =>
                {
                    ImagesFoundCount = summary.FilesFound.Images;
                    DocsFoundCount = summary.FilesFound.Documents;
                    StagedOcrCount = summary.Staged.Ocr;
                    StagedCaptionCount = summary.Staged.OcrCaption;
                    StoredPhotosCount = summary.Stored.Photos;
                    StoredDocsCount = summary.Stored.DocsOcr;
                    IsSummaryLoading = false;
                });
            }
        }
        catch (Exception ex)
        {
            LoggerService.Warn($"Summary query error: {ex.Message}");
            Dispatcher.UIThread.Post(() => IsSummaryLoading = false);
        }
    }

    private void ClearSummaryCounts()
    {
        ImagesFoundCount = 0;
        DocsFoundCount = 0;
        StagedOcrCount = 0;
        StagedCaptionCount = 0;
        StoredPhotosCount = 0;
        StoredDocsCount = 0;
    }

    private async Task RefreshStatusAsync()
    {
        if (_apiClient == null) return;
        try
        {
            var status = await _apiClient.GetStatusAsync();
            Dispatcher.UIThread.Post(() =>
            {
                DeviceName = status.DeviceName;
                VramAllocatedMb = status.Vram.AllocatedMb;
                VramTotalMb = status.Vram.TotalMb > 0 ? status.Vram.TotalMb : 4096;
                VramPercent = Math.Clamp((status.Vram.AllocatedMb / VramTotalMb) * 100.0, 0, 100);
                VramText = $"{status.Vram.AllocatedMb:F0} / {VramTotalMb:F0} MB";

                ResidentModels.Clear();
                foreach (var m in status.ResidentModels)
                {
                    ResidentModels.Add(m);
                }
            });
        }
        catch (Exception ex)
        {
            LoggerService.Warn($"Failed to refresh status: {ex.Message}");
        }
    }

    // =========================================================================
    // ACTION COMMANDS
    // =========================================================================
    [RelayCommand]
    private async Task ExecuteOcrAsync()
    {
        if (!IsFolderValid || IsJobRunning || _apiClient == null) return;
        IsJobRunning = true;
        IsOcrActive = true;
        ActiveJobType = "ocr";
        IsProgressVisible = true;
        ProgressPercent = 0;
        IsIndeterminate = true;
        StageTitle = "Starting OCR...";
        ResultStatus = "None";

        try
        {
            await _apiClient.TriggerOcrAsync(FolderInput);
        }
        catch (Exception ex)
        {
            OnJobLaunchFailed(ex.Message);
        }
    }

    [RelayCommand]
    private async Task ExecuteOcrCaptionAsync()
    {
        if (!IsFolderValid || IsJobRunning || _apiClient == null) return;
        IsJobRunning = true;
        IsOcrCaptionActive = true;
        ActiveJobType = "ocr_caption";
        IsProgressVisible = true;
        ProgressPercent = 0;
        IsIndeterminate = true;
        StageTitle = "Starting Florence-2 Caption + OCR...";
        ResultStatus = "None";

        try
        {
            await _apiClient.TriggerOcrCaptionAsync(FolderInput);
        }
        catch (Exception ex)
        {
            OnJobLaunchFailed(ex.Message);
        }
    }

    [RelayCommand]
    private async Task ExecuteStoreAsync()
    {
        if (!IsFolderValid || IsJobRunning || _apiClient == null) return;
        if (StagedOcrCount == 0 && StagedCaptionCount == 0)
        {
            ResultStatus = "Failed";
            ResultMessage = "Run OCR or OCR + Caption first. No staged data found.";
            return;
        }

        IsJobRunning = true;
        IsStoreActive = true;
        ActiveJobType = "store";
        IsProgressVisible = true;
        ProgressPercent = 0;
        IsIndeterminate = true;
        StageTitle = "Storing in Database...";
        ResultStatus = "None";

        try
        {
            await _apiClient.TriggerStoreAsync(FolderInput);
        }
        catch (Exception ex)
        {
            OnJobLaunchFailed(ex.Message);
        }
    }

    [RelayCommand]
    private void PromptDeleteModal()
    {
        if (!IsFolderValid || IsJobRunning) return;
        int totalStored = StoredPhotosCount + StoredDocsCount;
        DeleteModalText = $"Delete {totalStored} entries for '{Path.GetFileName(FolderInput)}' from the database?\n\nYour original image/document files are NEVER touched.";
        DeleteHoldProgress = 0;
        IsDeleteModalOpen = true;
    }

    [RelayCommand]
    private void CancelDeleteModal()
    {
        IsDeleteModalOpen = false;
        DeleteHoldProgress = 0;
    }

    [RelayCommand]
    private async Task ConfirmDeleteAsync()
    {
        IsDeleteModalOpen = false;
        if (!IsFolderValid || IsJobRunning || _apiClient == null) return;

        IsJobRunning = true;
        IsDeleteActive = true;
        ActiveJobType = "delete";
        IsProgressVisible = true;
        ProgressPercent = 0;
        IsIndeterminate = true;
        StageTitle = "Deleting from Database...";
        ResultStatus = "None";

        try
        {
            await _apiClient.TriggerDeleteAsync(FolderInput);
        }
        catch (Exception ex)
        {
            OnJobLaunchFailed(ex.Message);
        }
    }

    private void OnJobLaunchFailed(string message)
    {
        IsJobRunning = false;
        ResetActiveActionStates();
        ResultStatus = "Failed";
        ResultMessage = message;
    }

    [RelayCommand]
    private async Task CancelCurrentJobAsync()
    {
        if (!IsJobRunning || _apiClient == null) return;
        await _apiClient.CancelJobAsync();
    }

    [RelayCommand]
    private async Task RestartEngineAsync()
    {
        await StartBackendAndConnectAsync();
    }

    [RelayCommand]
    private void CopyLogs()
    {
        string allLogs = string.Join(Environment.NewLine, Logs.Select(l => $"[{l.Timestamp}] [{l.Level}] {l.Message}"));
        // Copy to system clipboard handled in View code-behind
    }

    [RelayCommand]
    private void OpenLogsFolder()
    {
        try
        {
            string dir = SettingsService.LogsDirectory;
            Directory.CreateDirectory(dir);
            Process.Start(new ProcessStartInfo { FileName = dir, UseShellExecute = true });
        }
        catch
        {
        }
    }

    [RelayCommand]
    private void ToggleContextMenu()
    {
        if (IsContextMenuRegistered)
        {
            ContextMenuService.Unregister();
            IsContextMenuRegistered = false;
        }
        else
        {
            ContextMenuService.Register();
            IsContextMenuRegistered = true;
        }
    }

    // =========================================================================
    // NAVIGATION COMMANDS
    // =========================================================================
    [RelayCommand]
    private void ShowSearchTab()
    {
        SelectedTabIndex = 0;
    }

    [RelayCommand]
    private void ShowIndexTab()
    {
        SelectedTabIndex = 1;
    }

    [RelayCommand]
    private void SelectTab(object? param)
    {
        if (param is int i)
        {
            SelectedTabIndex = i;
        }
        else if (param != null && int.TryParse(param.ToString(), out int parsed))
        {
            SelectedTabIndex = parsed;
        }
    }

    [RelayCommand]
    private async Task ExecuteSearchAsync()
    {
        string query = SearchQuery?.Trim() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(query) || _apiClient == null || IsSearching) return;

        IsSearching = true;
        HasSearched = true;
        SearchBadge = "Searching database...";
        SearchStatusText = string.Empty;
        SearchResults.Clear();
        OnPropertyChanged(nameof(HasSearchResults));
        OnPropertyChanged(nameof(NoResultsFound));

        try
        {
            var response = await _apiClient.SearchAsync(query, mode: SearchMode, topK: 3);
            SearchBadge = response.Badge;
            SearchStatusText = response.Results.Count > 0
                ? $"Top {response.Results.Count} matches retrieved"
                : $"No matching items found for '{query}'";

            foreach (var r in response.Results)
            {
                var itemModel = new SearchResultItemModel
                {
                    Rank = r.Rank,
                    Path = r.Path,
                    Filename = r.Filename,
                    Score = r.Score,
                    Date = r.Date,
                    Caption = r.Caption,
                    OcrText = r.OcrText,
                    Snippets = r.Snippets,
                };

                // Load thumbnail in background
                _ = LoadThumbnailForItemAsync(itemModel);

                SearchResults.Add(itemModel);
            }
        }
        catch (Exception ex)
        {
            SearchBadge = "Search error";
            SearchStatusText = ex.Message;
            LoggerService.Error($"Search error: {ex.Message}");
        }
        finally
        {
            IsSearching = false;
            OnPropertyChanged(nameof(HasSearchResults));
            OnPropertyChanged(nameof(NoResultsFound));
        }
    }

    private async Task LoadThumbnailForItemAsync(SearchResultItemModel item)
    {
        try
        {
            if (File.Exists(item.Path))
            {
                string ext = Path.GetExtension(item.Path).ToLowerInvariant();
                if (ext is ".jpg" or ".jpeg" or ".png" or ".webp" or ".bmp")
                {
                    await using var fileStream = File.OpenRead(item.Path);
                    var bmp = await Task.Run(() => Avalonia.Media.Imaging.Bitmap.DecodeToWidth(fileStream, 320));
                    await Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        item.Thumbnail = bmp;
                        item.HasThumbnail = true;
                    });
                    return;
                }
            }

            if (_apiClient != null)
            {
                var bytes = await _apiClient.GetThumbnailBytesAsync(item.Path);
                if (bytes != null && bytes.Length > 0)
                {
                    using var ms = new MemoryStream(bytes);
                    var bmp = new Avalonia.Media.Imaging.Bitmap(ms);
                    await Dispatcher.UIThread.InvokeAsync(() =>
                    {
                        item.Thumbnail = bmp;
                        item.HasThumbnail = true;
                    });
                }
            }
        }
        catch (Exception ex)
        {
            LoggerService.Warn($"Failed to load thumbnail for {item.Filename}: {ex.Message}");
        }
    }

    [RelayCommand]
    private async Task SetQuickSearchAsync(string suggestion)
    {
        SearchQuery = suggestion;
        await ExecuteSearchAsync();
    }

    [RelayCommand]
    private void OpenFile(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = path,
                    UseShellExecute = true,
                });
            }
        }
        catch (Exception ex)
        {
            LoggerService.Error($"Failed to open file: {ex.Message}");
        }
    }

    [RelayCommand]
    private async Task RevealFileAsync(string path)
    {
        try
        {
            if (_apiClient != null)
            {
                await _apiClient.RevealFileAsync(path);
            }
            else if (File.Exists(path))
            {
                Process.Start(new ProcessStartInfo
                {
                    FileName = "explorer.exe",
                    Arguments = $"/select,\"{path}\"",
                    UseShellExecute = true,
                });
            }
        }
        catch (Exception ex)
        {
            LoggerService.Error($"Failed to reveal file: {ex.Message}");
        }
    }

    public async Task CleanupAsync()
    {
        _webSocketClient?.Dispose();
        if (_apiClient != null)
        {
            await _apiClient.ShutdownAsync();
            _apiClient.Dispose();
        }
        _backendManager?.Dispose();
    }
}
