using System;
using System.Linq;
using Avalonia;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml;
using Avalonia.Threading;
using EmbedEye.Gui.ViewModels;
using EmbedEye.Gui.Views;

namespace EmbedEye.Gui;

public partial class App : Application
{
    public override void Initialize()
    {
        AvaloniaXamlLoader.Load(this);
    }

    public override void OnFrameworkInitializationCompleted()
    {
        if (ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
        {
            // Read initial folder argument if provided
            string? initialFolder = null;
            if (desktop.Args != null)
            {
                foreach (var arg in desktop.Args)
                {
                    if (!arg.StartsWith("--", StringComparison.OrdinalIgnoreCase))
                    {
                        initialFolder = arg;
                        break;
                    }
                }
            }

            var vm = new MainWindowViewModel();
            var window = new MainWindow
            {
                DataContext = vm,
            };

            desktop.MainWindow = window;

            // Hook single-instance pipe listener to forward folder into running window
            if (Program.SingleInstance != null)
            {
                Program.SingleInstance.FolderArgumentReceived += folder =>
                {
                    Dispatcher.UIThread.Post(() =>
                    {
                        if (window.WindowState == Avalonia.Controls.WindowState.Minimized)
                        {
                            window.WindowState = Avalonia.Controls.WindowState.Normal;
                        }
                        window.Activate();
                        window.Focus();
                        vm.FolderInput = folder;
                    });
                };
            }

            // Start backend engine asynchronously
            Dispatcher.UIThread.Post(async () =>
            {
                await vm.InitializeAsync(initialFolder);
            });
        }

        base.OnFrameworkInitializationCompleted();
    }
}