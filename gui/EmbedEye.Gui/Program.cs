using System;
using System.IO;
using System.Linq;
using Avalonia;
using Avalonia.Rendering;
using EmbedEye.Gui.Services;

namespace EmbedEye.Gui;

sealed class Program
{
    private static SingleInstanceService? _singleInstanceService;

    [STAThread]
    public static void Main(string[] args)
    {
        // 1. Check for command-line context menu registration flags
        if (args.Contains("--register-context-menu", StringComparer.OrdinalIgnoreCase))
        {
            bool ok = ContextMenuService.Register();
            Console.WriteLine(ok ? "Context menu registered successfully." : "Failed to register context menu.");
            return;
        }

        if (args.Contains("--unregister-context-menu", StringComparer.OrdinalIgnoreCase))
        {
            bool ok = ContextMenuService.Unregister();
            Console.WriteLine(ok ? "Context menu unregistered successfully." : "Failed to unregister context menu.");
            return;
        }

        // 2. Single instance check: acquire Mutex or forward argument to primary window
        _singleInstanceService = new SingleInstanceService();
        if (!_singleInstanceService.TryAcquire(args))
        {
            // Another instance is already running; folder argument was forwarded
            return;
        }

        // 3. Build and launch Avalonia application
        var builder = BuildAvaloniaApp(args);
        builder.StartWithClassicDesktopLifetime(args);

        _singleInstanceService.Dispose();
    }

    public static SingleInstanceService? SingleInstance => _singleInstanceService;

    public static AppBuilder BuildAvaloniaApp(string[]? args = null)
    {
        var builder = AppBuilder.Configure<App>()
            .UsePlatformDetect()
            .WithInterFont()
            .LogToTrace();

        // Optional software rendering fallback flag
        if (args != null && args.Contains("--software-render", StringComparer.OrdinalIgnoreCase))
        {
            builder.With(new Win32PlatformOptions
            {
                RenderingMode = new[] { Win32RenderingMode.Software }
            });
        }

        return builder;
    }

    // Designer fallback
    public static AppBuilder BuildAvaloniaApp() => BuildAvaloniaApp(null);
}
