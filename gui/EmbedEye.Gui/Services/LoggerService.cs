using System;
using System.IO;

namespace EmbedEye.Gui.Services;

public static class LoggerService
{
    private static readonly object FileLock = new();

    public static void Log(string level, string message)
    {
        try
        {
            string dir = SettingsService.LogsDirectory;
            Directory.CreateDirectory(dir);
            string logFile = Path.Combine(dir, $"gui_{DateTime.Now:yyyyMMdd}.log");
            string line = $"{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff} [{level.ToUpper()}] {message}{Environment.NewLine}";

            lock (FileLock)
            {
                File.AppendAllText(logFile, line);
            }
        }
        catch
        {
            // Silently ignore logging failures
        }
    }

    public static void Info(string message) => Log("INFO", message);
    public static void Warn(string message) => Log("WARN", message);
    public static void Error(string message, Exception? ex = null)
    {
        string fullMsg = ex != null ? $"{message} - {ex}" : message;
        Log("ERROR", fullMsg);
    }
}
