using System;
using System.Diagnostics;
using Microsoft.Win32;

namespace EmbedEye.Gui.Services;

public static class ContextMenuService
{
    private const string DirectoryKey = @"Software\Classes\Directory\shell\EmbedEye";
    private const string BackgroundKey = @"Software\Classes\Directory\Background\shell\EmbedEye";

    public static bool IsRegistered()
    {
        if (!OperatingSystem.IsWindows()) return false;
        try
        {
            using var key = Registry.CurrentUser.OpenSubKey(DirectoryKey);
            return key != null;
        }
        catch
        {
            return false;
        }
    }

    public static bool Register()
    {
        if (!OperatingSystem.IsWindows()) return false;

        try
        {
            string exePath = Process.GetCurrentProcess().MainModule?.FileName ?? Environment.ProcessPath ?? "";
            if (string.IsNullOrWhiteSpace(exePath)) return false;

            // 1. Directory shell key
            using (var key = Registry.CurrentUser.CreateSubKey(DirectoryKey))
            {
                if (key != null)
                {
                    key.SetValue("", "Open with EmbedEye");
                    key.SetValue("Icon", $"\"{exePath}\"");
                    using var cmdKey = key.CreateSubKey("command");
                    cmdKey?.SetValue("", $"\"{exePath}\" \"%1\"");
                }
            }

            // 2. Directory background shell key
            using (var key = Registry.CurrentUser.CreateSubKey(BackgroundKey))
            {
                if (key != null)
                {
                    key.SetValue("", "Open with EmbedEye");
                    key.SetValue("Icon", $"\"{exePath}\"");
                    using var cmdKey = key.CreateSubKey("command");
                    cmdKey?.SetValue("", $"\"{exePath}\" \"%V\"");
                }
            }

            LoggerService.Info("Successfully registered HKCU Windows Explorer context menu.");
            return true;
        }
        catch (Exception ex)
        {
            LoggerService.Error("Failed to register context menu", ex);
            return false;
        }
    }

    public static bool Unregister()
    {
        if (!OperatingSystem.IsWindows()) return false;

        try
        {
            Registry.CurrentUser.DeleteSubKeyTree(DirectoryKey, false);
            Registry.CurrentUser.DeleteSubKeyTree(BackgroundKey, false);
            LoggerService.Info("Successfully unregistered HKCU Windows Explorer context menu.");
            return true;
        }
        catch (Exception ex)
        {
            LoggerService.Error("Failed to unregister context menu", ex);
            return false;
        }
    }
}
