nnoremap <C-k> I# <Esc>
" ============================================
" ПЛАГИНЫ
" ============================================
call plug#begin()
  " Oceanic Next - цветовая схема
  Plug 'mhartington/oceanic-next'
  
  " COC - автодополнение
  Plug 'neoclide/coc.nvim', {'branch': 'release'}
call plug#end()

" ============================================
" ПОДКЛЮЧЕНИЕ ФАЙЛА С ТЕМАМИ
" ============================================
if filereadable(expand("~/.vim/themes.vim"))
  source ~/.vim/themes.vim
else
  echo "Файл ~/.vim/themes.vim не найден"
endif

" ============================================
" ОСНОВНЫЕ НАСТРОЙКИ
" ============================================

" Кодировка UTF-8
set encoding=utf8

" Отключение совместимости с vi
set nocompatible

" Игнорировать регистр при поиске
set ignorecase

" Не игнорировать регистр, если в паттерне есть большие буквы
set smartcase

" Подсвечивать найденный паттерн
set hlsearch

" Интерактивный поиск
set incsearch

" Размер табов - 2 (для общих файлов)
set tabstop=2
set softtabstop=2
set shiftwidth=2

" Превратить табы в пробелы
set expandtab

" Таб перед строкой будет вставлять количество пробелов определённое в shiftwidth
set smarttab

" Копировать отступ на новой строке
set autoindent
set smartindent

" Показывать номера строк
set number

" Относительные номера строк
set relativenumber

" Автокомплиты в командной строке
set wildmode=longest,list

" Подсветка синтаксиса
syntax on

" Разрешить использование мыши
set mouse=a

" Использовать системный буфер обмена
set clipboard=unnamedplus

" Быстрый скроллинг
set ttyfast

" Курсор во время скроллинга будет всегда в середине экрана
set so=30

" ============================================
" НАСТРОЙКИ ДЛЯ PYTHON
" ============================================

" Отступы для Python
autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab autoindent

" ============================================
" КОМАНДЫ И ГОРЯЧИЕ КЛАВИШИ
" ============================================

" Запуск Python файла (Ctrl+L)
nnoremap <C-l> :w<CR>:!clear && python3 %<CR>

" Раскладка клавиатуры для русского языка
set langmap=ФИСВУАПРШОЛДЬТЩЗЙКЫЕГМЦЧНЯ;ABCDEFGHIJKLMNOPQRSTUVWXYZ,фисвуапршолдьтщзйкыегмцчня;abcdefghijklmnopqrstuvwxyz

" Переключение цветовых схем (F4)
map <F4> :execute RotateColor()<CR>:colorscheme<CR>

let b:colindex=0
function! RotateColor()
  let y = -1
  while y == -1
    let colstring="#darkblue#default#industry#murphy#peachpuff#torte#"
    let x=match(colstring,"#",b:colindex)
    let y=match(colstring,"#",x+1)
    let b:colindex=x+1
    if y == -1
      let b:colindex=0
    else
      let str=strpart(colstring,x+1,y-x-1)
      return ":colorscheme " .str
    endif
  endwhile
endfunction

" Автокомплиты через Tab
inoremap <expr> <Tab> coc#pum#visible() ? coc#pum#confirm() : "\<Tab>"

" ============================================
" КОПИРОВАНИЕ В СИСТЕМНЫЙ БУФЕР
" ============================================

" Копировать выделенное в системный буфер (Ctrl+C)
vnoremap <C-c> "xy:call system('xclip -selection clipboard', getreg('x'))<CR>

" Копировать текущую строку (Ctrl+Y)
nnoremap <C-y> "xyy:call system('xclip -selection clipboard', getreg('x'))<CR>

" Копировать весь файл (Ctrl+Shift+C)
nnoremap <C-S-c> ggVG"xy:call system('xclip -selection clipboard', getreg('x'))<CR>

" ============================================
" АКТИВАЦИЯ ТЕМЫ ПО УМОЛЧАНИЮ
" ============================================

" Принудительно устанавливаем Oceanic Next как тему по умолчанию
colorscheme OceanicNext
