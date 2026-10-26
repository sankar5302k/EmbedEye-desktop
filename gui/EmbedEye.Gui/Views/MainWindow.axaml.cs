using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using EmbedEye.Gui.ViewModels;

namespace EmbedEye.Gui.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DropEvent, OnDrop);

        KeyDown += OnWindowKeyDown;
        Closing += OnWindowClosing;
    }

    private MainWindowViewModel? ViewModel => DataContext as MainWindowViewModel;

    private async void OnBrowseFolderClicked(object? sender, RoutedEventArgs e)
    {
        var topLevel = GetTopLevel(this);
        if (topLevel?.StorageProvider == null) return;

        var folders = await topLevel.StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select Image or Document Folder to Index",
            AllowMultiple = false
        });

        if (folders.Count > 0 && ViewModel != null)
        {
            string? localPath = folders[0].TryGetLocalPath();
            if (string.IsNullOrEmpty(localPath) && folders[0].Path != null)
            {
                localPath = folders[0].Path.LocalPath;
            }
            if (!string.IsNullOrEmpty(localPath))
            {
                ViewModel.FolderInput = localPath;
            }
        }
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        if (e.Data.Contains(DataFormats.Files))
        {
            e.DragEffects = DragDropEffects.Copy;
        }
        else
        {
            e.DragEffects = DragDropEffects.None;
        }
    }

    private void OnDrop(object? sender, DragEventArgs e)
    {
        if (e.Data.Contains(DataFormats.Files) && ViewModel != null)
        {
            var files = e.Data.GetFiles();
            if (files != null)
            {
                var first = files.FirstOrDefault();
                if (first != null)
                {
                    string? path = first.TryGetLocalPath();
                    if (!string.IsNullOrEmpty(path))
                    {
                        if (Directory.Exists(path))
                        {
                            ViewModel.FolderInput = path;
                        }
                        else if (File.Exists(path))
                        {
                            string? dir = Path.GetDirectoryName(path);
                            if (!string.IsNullOrEmpty(dir))
                            {
                                ViewModel.FolderInput = dir;
                            }
                        }
                    }
                }
            }
        }
    }

    private async void OnCopyLogsClicked(object? sender, RoutedEventArgs e)
    {
        if (ViewModel == null) return;
        string allLogs = string.Join(Environment.NewLine,
            ViewModel.Logs.Select(l => $"[{l.Timestamp}] [{l.Level}] {l.Message}"));

        var clipboard = GetTopLevel(this)?.Clipboard;
        if (clipboard != null && !string.IsNullOrEmpty(allLogs))
        {
            await clipboard.SetTextAsync(allLogs);
        }
    }

    private void OnSearchTextBoxKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && ViewModel != null && !ViewModel.IsSearching)
        {
            ViewModel.ExecuteSearchCommand.Execute(null);
            e.Handled = true;
        }
    }

    private void OnWindowKeyDown(object? sender, KeyEventArgs e)
    {
        if (ViewModel == null) return;

        // Esc: Cancel active job or close modal
        if (e.Key == Key.Escape)
        {
            if (ViewModel.IsDeleteModalOpen)
            {
                ViewModel.CancelDeleteModalCommand.Execute(null);
                e.Handled = true;
            }
            else if (ViewModel.IsJobRunning)
            {
                ViewModel.CancelCurrentJobCommand.Execute(null);
                e.Handled = true;
            }
        }
    }

    private async void OnWindowClosing(object? sender, WindowClosingEventArgs e)
    {
        if (ViewModel != null)
        {
            await ViewModel.CleanupAsync();
        }
    }
}