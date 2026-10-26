using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace EmbedEye.Gui.Services;

public class AppSettings
{
    public string PythonPath { get; set; } = string.Empty;
    public string RepoRoot { get; set; } = string.Empty;
    public bool ReduceAnimations { get; set; } = false;
}

public sealed class SettingsService
{
    private static readonly string SettingsDirectory = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "EmbedEye");

    private static readonly string SettingsFilePath = Path.Combine(SettingsDirectory, "settings.json");

    public AppSettings Current { get; private set; }

    public SettingsService()
    {
        Current = LoadSettings();
    }

    public static string LogsDirectory => Path.Combine(SettingsDirectory, "logs");

    private AppSettings LoadSettings()
    {
        AppSettings? settings = null;
        try
        {
            if (File.Exists(SettingsFilePath))
            {
                string json = File.ReadAllText(SettingsFilePath);
                settings = JsonSerializer.Deserialize<AppSettings>(json);
            }
        }
        catch
        {
            // Fall back to auto-detection
        }

        settings ??= new AppSettings();

        // Auto-detect Repo Root if empty or invalid
        if (string.IsNullOrWhiteSpace(settings.RepoRoot) || !Directory.Exists(settings.RepoRoot))
        {
            settings.RepoRoot = DiscoverRepoRoot();
        }

        // Auto-detect Python path if empty or invalid
        if (string.IsNullOrWhiteSpace(settings.PythonPath) || !File.Exists(settings.PythonPath))
        {
            settings.PythonPath = DiscoverPythonPath(settings.RepoRoot);
        }

        // Check Windows system animation setting if not explicitly overridden
        if (OperatingSystem.IsWindows() && !File.Exists(SettingsFilePath))
        {
            settings.ReduceAnimations = !IsWindowsAnimationEnabled();
        }

        SaveSettings(settings);
        return settings;
    }

    public void Save()
    {
        SaveSettings(Current);
    }

    private void SaveSettings(AppSettings settings)
    {
        try
        {
            Directory.CreateDirectory(SettingsDirectory);
            string json = JsonSerializer.Serialize(settings, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(SettingsFilePath, json);
        }
        catch
        {
            // Ignore write failures in restricted environments
        }
    }

    private static string DiscoverRepoRoot()
    {
        string current = AppContext.BaseDirectory;
        var dir = new DirectoryInfo(current);
        while (dir != null)
        {
            if (File.Exists(Path.Combine(dir.FullName, "requirements.txt")) &&
                Directory.Exists(Path.Combine(dir.FullName, "src")))
            {
                return dir.FullName;
            }
            dir = dir.Parent;
        }

        // Fallback to current working directory
        return Directory.GetCurrentDirectory();
    }

    private static string DiscoverPythonPath(string repoRoot)
    {
        // Check venv pythonw.exe first (hidden background execution)
        string[] candidates = new[]
        {
            Path.Combine(repoRoot, "venv", "Scripts", "pythonw.exe"),
            Path.Combine(repoRoot, "venv", "Scripts", "python.exe"),
            Path.Combine(repoRoot, ".venv", "Scripts", "pythonw.exe"),
            Path.Combine(repoRoot, ".venv", "Scripts", "python.exe"),
        };

        foreach (var c in candidates)
        {
            if (File.Exists(c))
                return c;
        }

        return "python.exe";
    }

    public static bool IsWindowsAnimationEnabled()
    {
        if (!OperatingSystem.IsWindows())
            return true;

        try
        {
            const uint SPI_GETCLIENTAREAANIMATION = 0x1042;
            bool enabled = true;
            if (SystemParametersInfo(SPI_GETCLIENTAREAANIMATION, 0, ref enabled, 0))
            {
                return enabled;
            }
        }
        catch
        {
            // Default to true
        }

        return true;
    }

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SystemParametersInfo(uint uiAction, uint uiParam, ref bool pvParam, uint fWinIni);
}
