// to integrate another syntax highlighter or add some custom logic to your theme

// 确保你的 JavaScript 代码在正确的时机运行，尤其是在页面完全加载之后。
// 通过订阅 document$，你可以在页面加载完成时初始化一些第三方库或者执行其他需要在页面加载完成后运行的代码。
document$.subscribe(function() {
    console.log("Initialize third-party libraries here")
  })